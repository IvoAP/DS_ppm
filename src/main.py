from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	cohen_kappa_score,
	f1_score,
	matthews_corrcoef,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42
TARGET_COLUMN = "income"
SPLITS = [0.6, 0.7, 0.8]
DEFECTIVE_NODES = [4, 5, 10]
CLUSTERS_PER_NODE = 5


@dataclass
class LinearAnonTransform:
	mean: np.ndarray
	matrix: np.ndarray


@dataclass
class ClusterAnonModel:
	kmeans: KMeans
	transforms: dict[int, LinearAnonTransform]


def fit_linear_anonymizer(data: np.ndarray, rng: np.random.Generator) -> LinearAnonTransform:
	n_samples, n_features = data.shape
	if n_samples < 2:
		mean = np.array(np.mean(data, axis=0).T)
		return LinearAnonTransform(mean=mean, matrix=np.eye(n_features, dtype=np.float64))

	mean = np.array(np.mean(data, axis=0).T)
	data_centered = data - mean
	cov_matrix = np.cov(data_centered, rowvar=False)
	cov_matrix = np.atleast_2d(cov_matrix)
	cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
	cov_matrix = cov_matrix + (1e-8 * np.eye(cov_matrix.shape[0]))

	evals, evecs = la.eigh(cov_matrix)
	idx = np.argsort(evals)[::-1]
	evecs = evecs[:, idx]

	randomized_evecs = evecs.copy().T
	for i in range(len(randomized_evecs)):
		rng.shuffle(randomized_evecs[i])
	randomized_evecs = randomized_evecs.T

	matrix = evecs @ randomized_evecs.T
	return LinearAnonTransform(mean=mean, matrix=matrix)


def transform_with_linear_anonymizer(data: np.ndarray, model: LinearAnonTransform) -> np.ndarray:
	centered = data - model.mean
	transformed = centered @ model.matrix
	return transformed + model.mean


def fit_cluster_anonymizer(data: np.ndarray, clusters: int, rng: np.random.Generator) -> ClusterAnonModel:
	k_eff = max(1, min(clusters, len(data)))
	kmeans = KMeans(n_clusters=k_eff, random_state=RANDOM_SEED, n_init=10)
	labels = kmeans.fit_predict(data)

	transforms: dict[int, LinearAnonTransform] = {}
	for cluster_id in range(k_eff):
		cluster_data = data[labels == cluster_id]
		if len(cluster_data) == 0:
			continue
		transforms[cluster_id] = fit_linear_anonymizer(cluster_data, rng)

	return ClusterAnonModel(kmeans=kmeans, transforms=transforms)


def transform_with_cluster_anonymizer(data: np.ndarray, model: ClusterAnonModel) -> np.ndarray:
	labels = model.kmeans.predict(data)
	transformed = np.empty_like(data)

	for cluster_id, anon_model in model.transforms.items():
		mask = labels == cluster_id
		if np.any(mask):
			transformed[mask] = transform_with_linear_anonymizer(data[mask], anon_model)

	return transformed


def preprocess_adult_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
	df = pd.read_csv(csv_path)
	df = df.replace("?", "Unknown")

	y = (df[TARGET_COLUMN].str.strip() == ">50K").astype(int).to_numpy()
	X = df.drop(columns=[TARGET_COLUMN])
	X_encoded = pd.get_dummies(X, drop_first=False)

	return X_encoded.to_numpy(dtype=np.float64), y


def split_indices_for_nodes(n_samples: int, total_nodes: int) -> list[np.ndarray]:
	indices = np.arange(n_samples)
	return [chunk for chunk in np.array_split(indices, total_nodes) if len(chunk) > 0]


def byzantine_probabilities(average_honest_probs: np.ndarray) -> np.ndarray:
	return 1.0 - average_honest_probs


def trimmed_mean_probabilities(node_probabilities: np.ndarray, defective_nodes: int) -> np.ndarray:
	n_nodes = node_probabilities.shape[0]
	if 2 * defective_nodes >= n_nodes:
		raise ValueError("trimmed_mean requires n_nodes > 2 * defective_nodes")

	sorted_probs = np.sort(node_probabilities, axis=0)
	trimmed = sorted_probs[defective_nodes : n_nodes - defective_nodes]
	return trimmed.mean(axis=0)


def krum_probabilities(node_probabilities: np.ndarray, defective_nodes: int) -> np.ndarray:
	n_nodes = node_probabilities.shape[0]
	neighbors = n_nodes - defective_nodes - 2
	if neighbors <= 0:
		raise ValueError("krum requires n_nodes - defective_nodes - 2 > 0")

	pairwise = np.linalg.norm(
		node_probabilities[:, None, :] - node_probabilities[None, :, :],
		axis=2,
	)

	scores = np.empty(n_nodes, dtype=np.float64)
	for i in range(n_nodes):
		distances = np.sort(np.delete(pairwise[i], i))
		scores[i] = np.sum(distances[:neighbors])

	best_idx = int(np.argmin(scores))
	return node_probabilities[best_idx]


def krum_scores(node_probabilities: np.ndarray, defective_nodes: int) -> np.ndarray:
	n_nodes = node_probabilities.shape[0]
	neighbors = n_nodes - defective_nodes - 2
	if neighbors <= 0:
		raise ValueError("Krum-family methods require n_nodes - defective_nodes - 2 > 0")

	pairwise = np.linalg.norm(
		node_probabilities[:, None, :] - node_probabilities[None, :, :],
		axis=2,
	)

	scores = np.empty(n_nodes, dtype=np.float64)
	for i in range(n_nodes):
		distances = np.sort(np.delete(pairwise[i], i))
		scores[i] = np.sum(distances[:neighbors])

	return scores


def multi_krum_probabilities(node_probabilities: np.ndarray, defective_nodes: int) -> np.ndarray:
	n_nodes = node_probabilities.shape[0]
	m = n_nodes - defective_nodes
	if m <= 0:
		raise ValueError("multi_krum requires n_nodes > defective_nodes")

	scores = krum_scores(node_probabilities, defective_nodes)
	selected = np.argsort(scores)[:m]
	return node_probabilities[selected].mean(axis=0)


def bulyan_probabilities(node_probabilities: np.ndarray, defective_nodes: int) -> np.ndarray:
	n_nodes = node_probabilities.shape[0]
	beta = n_nodes - (2 * defective_nodes)
	if beta <= 0:
		raise ValueError("bulyan requires n_nodes > 2 * defective_nodes")

	# Bulyan's second stage needs at least n > 4f to keep n-4f coordinates after trimming.
	# With n=3f+1 this condition is not satisfied, so we use the closest practical variant:
	# candidate selection via iterative Krum + adaptive trimmed mean over candidates.
	remaining = node_probabilities.copy()
	selected: list[np.ndarray] = []

	while len(selected) < beta:
		if remaining.shape[0] - defective_nodes - 2 <= 0:
			break

		scores = krum_scores(remaining, defective_nodes)
		best_idx = int(np.argmin(scores))
		selected.append(remaining[best_idx])
		remaining = np.delete(remaining, best_idx, axis=0)

	if not selected:
		raise ValueError("bulyan could not select any candidate vectors")

	selected_array = np.vstack(selected)
	adaptive_trim = min(defective_nodes, max(0, (selected_array.shape[0] - 1) // 2))
	if adaptive_trim == 0:
		return selected_array.mean(axis=0)

	sorted_probs = np.sort(selected_array, axis=0)
	trimmed = sorted_probs[adaptive_trim : selected_array.shape[0] - adaptive_trim]
	if trimmed.shape[0] == 0:
		return selected_array.mean(axis=0)
	return trimmed.mean(axis=0)


def aggregate_probabilities(node_probabilities: np.ndarray, method: str, defective_nodes: int) -> np.ndarray:
	if method == "mean":
		return node_probabilities.mean(axis=0)
	if method == "median":
		return np.median(node_probabilities, axis=0)
	if method == "trimmed_mean":
		return trimmed_mean_probabilities(node_probabilities, defective_nodes)
	if method == "krum":
		return krum_probabilities(node_probabilities, defective_nodes)
	if method == "multi_krum":
		return multi_krum_probabilities(node_probabilities, defective_nodes)
	if method == "bulyan":
		return bulyan_probabilities(node_probabilities, defective_nodes)
	raise ValueError(f"Unknown aggregation method: {method}")


def run_simulation(
	X: np.ndarray,
	y: np.ndarray,
	train_ratio: float,
	defective_nodes: int,
	clusters_per_node: int,
	rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
	total_nodes = 3 * defective_nodes + 1

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		train_size=train_ratio,
		random_state=RANDOM_SEED,
		stratify=y,
	)

	node_splits = split_indices_for_nodes(len(X_train), total_nodes)
	if len(node_splits) < total_nodes:
		raise ValueError("Dataset split produced fewer non-empty shards than expected nodes.")

	honest_nodes = total_nodes - defective_nodes
	honest_probabilities = []

	for node_id in range(total_nodes):
		if node_id >= honest_nodes:
			continue

		node_indices = node_splits[node_id]
		X_node = X_train[node_indices]
		y_node = y_train[node_indices]

		anon_model = fit_cluster_anonymizer(X_node, clusters_per_node, rng)
		X_node_anon = transform_with_cluster_anonymizer(X_node, anon_model)
		X_test_anon = transform_with_cluster_anonymizer(X_test, anon_model)

		clf = LogisticRegression(
			max_iter=2000,
			solver="liblinear",
			random_state=RANDOM_SEED,
		)
		clf.fit(X_node_anon, y_node)

		probs = clf.predict_proba(X_test_anon)[:, 1]
		honest_probabilities.append(probs)

	honest_probabilities_array = np.vstack(honest_probabilities)
	avg_honest_probs = honest_probabilities_array.mean(axis=0)

	byzantine_probs = [byzantine_probabilities(avg_honest_probs) for _ in range(defective_nodes)]
	all_node_probabilities = np.vstack([honest_probabilities_array, np.vstack(byzantine_probs)])

	rows: list[dict[str, float | int | str]] = []
	test_ratio = int(round((1 - train_ratio) * 100))
	for method in ["mean", "median", "trimmed_mean", "krum", "multi_krum", "bulyan"]:
		t0 = time.perf_counter()
		agg_probs = aggregate_probabilities(all_node_probabilities, method, defective_nodes)
		y_pred = (agg_probs >= 0.5).astype(int)
		elapsed = time.perf_counter() - t0

		rows.append(
			{
				"defective_nodes": defective_nodes,
				"total_nodes": total_nodes,
				"dataset_split": f"{int(round(train_ratio * 100))}/{test_ratio}",
				"aggregation_method": method,
				"accuracy": float(accuracy_score(y_test, y_pred)),
				"f1": float(f1_score(y_test, y_pred)),
				"precision": float(precision_score(y_test, y_pred, zero_division=0)),
				"recall": float(recall_score(y_test, y_pred, zero_division=0)),
				"cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
				"roc_auc": float(roc_auc_score(y_test, agg_probs)),
				"mcc": float(matthews_corrcoef(y_test, y_pred)),
				"time_seconds": float(elapsed),
			}
		)

	return rows


def main() -> None:
	root = Path(__file__).resolve().parents[1]
	csv_path = root / "data" / "adult.csv"
	output_path = root / "data" / "byzantine_results.csv"

	rng = np.random.default_rng(RANDOM_SEED)
	X, y = preprocess_adult_dataset(csv_path)

	results: list[dict[str, float | int | str]] = []
	for split_ratio in SPLITS:
		for defective in DEFECTIVE_NODES:
			results.extend(
				run_simulation(
					X=X,
					y=y,
					train_ratio=split_ratio,
					defective_nodes=defective,
					clusters_per_node=CLUSTERS_PER_NODE,
					rng=rng,
				)
			)

	results_df = pd.DataFrame(results)
	results_df.to_csv(output_path, index=False)

	print(f"Results saved to: {output_path}")
	print(results_df)


if __name__ == "__main__":
	main()

# DS_ppm

Distributed privacy-preserving machine learning simulation with Byzantine fault tolerance experiments on the Adult (Census Income) dataset.

## Project Overview

This project simulates a distributed learning setting where:

- each node receives a shard of training data,
- each node applies local anonymization,
- honest nodes train local classifiers,
- Byzantine nodes send malicious outputs,
- a central coordinator aggregates node outputs with different robust strategies.

The goal is to evaluate how faulty/adversarial nodes affect predictive performance and how robust aggregation algorithms can mitigate this impact.

## Implemented Components

### 1) Local Anonymization

The anonymization pipeline is cluster-based and PCA-like:

- data is split into local clusters,
- covariance eigenspace is computed per cluster,
- eigenvector structure is randomized,
- transformed data is mapped back to original dimensionality.

Numerical safeguards are included for tiny clusters to prevent NaN/Inf failures.

### 2) Distributed Byzantine Simulation

For each experiment:

- number of defective nodes: `f in {4, 5, 10}`,
- total nodes follow `n = 3f + 1`,
- train/test splits: `60/40`, `70/30`, `80/20`.

Byzantine nodes are simulated by adversarial probability inversion (`1 - p`).

### 3) Aggregation Methods

The coordinator compares multiple aggregation rules:

- `mean`
- `median`
- `trimmed_mean`
- `krum`
- `multi_krum`
- `bulyan` (practical variant for this node regime)

## Metrics

Each experiment logs:

- `accuracy`
- `f1`
- `precision`
- `recall`
- `cohen_kappa`
- `roc_auc`
- `mcc`
- `time_seconds`

## Running the Simulation

From the project root:

```bash
uv run src/main.py
```

This generates/updates the experiment output CSV.

## Output

Main result file:

- `data/byzantine_results.csv`

Columns:

- `defective_nodes`
- `total_nodes`
- `dataset_split`
- `aggregation_method`
- `accuracy`
- `f1`
- `precision`
- `recall`
- `cohen_kappa`
- `roc_auc`
- `mcc`
- `time_seconds`

Expected experiment grid size:

- `3` defective-node settings × `3` data splits × `6` aggregators = `54` rows.

## Repository Structure

```text
pyproject.toml
README.md
data/
	adult.csv
	byzantine_results.csv
src/
	anon.py
	main.py
docs/
	briefing.txt
```

## Notes

- The project is designed for reproducible experimentation and comparison of robust aggregation methods.
- If you change the Byzantine attack model, method rankings may change significantly.
- `main.py` is the end-to-end execution entry point for all experiments.
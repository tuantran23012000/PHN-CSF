# Multi Objective Optimization experiments
## Usage

Custom Objective functions in `problems/get_problem.py`

Available multi-objective optimization problems:

| command-line option  | Num of variables      | Num of objectives | Objective function | Pareto-optimal|
|----------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| `--problem ex1`              | 1           |2|convex|convex|
| `--problem ex2`              | 2       | 2 | convex | convex|
| `--problem ex3`             | 3 | 3 | convex | convex|
| `--problem ex4`           |    2         | 2 | convex | convex
| `--problem ZDT1`         |       30      |  2  | non-convex | convex |
| `--problem ZDT2`         |       30        |  2  | non-convex | non-convex |
| `--problem DTLZ2`          |       10         |  3  |non-convex | non-convex |
---
Custom Scalarization functions in `tools/scalarization_function.py`

Available scalarization functions:

| command-line option  | Description                  |
|----------------------|------------------------------|
| `--solver LS`              | Linear Scalarization function          |
| `--solver Cheby`              | Chebyshev function        |
| `--solver Utility`             | Inverse Utility function |
| `--solver EPO`           |                 |
| `--solver KL`         | Kullback–Leibler divergence function             |
| `--solver Cauchy`         | Cauchy–Schwarz function              |
| `--solver CS`          | Cosine-Similarity function               |
| `--solver Log`          | Log function               |
| `--solver Prod`          | Product function               |
| `--solver AC`          | Augmented Chebyshev function              |
| `--solver MC`          | Modified Chebyshev function              |
| `--solver HVI`          | Hypervolume Indicator function              |
---

Custom Model architecture and hyper-parameters in `configs/`

## Training
```
python3 main.py --solver <solver type> -- problem <problem name> --mode train --model_type <transformer/mlp>
```
or
```
sh main_train.sh
```
## Evaluating
```
python3 main.py --solver <solver type> -- problem <problem name> --mode predict --model_type <transformer/mlp>
```
or
```
sh main_test.sh
```
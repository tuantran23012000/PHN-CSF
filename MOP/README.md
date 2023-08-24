# A Framework for Pareto Multi Task Learning with Completed Scalarization Functions
## Custom code
Custom Objective functions in `problems/get_problem.py`

Custom Scalarization functions in `tools/scalarization_function.py`

Custom Model architecture and hyper-parameters in `configs/`

## Training
```
python3 main.py --solver <solver type> -- problem <problem name> --mode train
```
## Evaluating
```
python3 main.py --solver <solver type> -- problem <problem name> --mode predict
```

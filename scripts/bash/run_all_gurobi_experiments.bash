python scripts/train.py --config-name cflp-medium lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi &
python scripts/train.py --config-name gisp-easy lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi &
python scripts/train.py --config-name mis-medium lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi &
python scripts/train.py --config-name mvc-medium lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi &
python scripts/train.py --config-name nnv-easy lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi &
python scripts/train.py --config-name ots-medium lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi &
python scripts/train.py --config-name facilities lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi &
python scripts/train.py --config-name mmcn-medium-bi lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi

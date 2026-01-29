# python scripts/train.py --config-name $1 lr=0.0001 model_type=lstm bound_output=True device=cuda &
# python scripts/train.py --config-name $1 lr=0.0001 model_type=feedforward bound_output=True device=cuda &
python scripts/train.py --config-name $1 lr=0.0001 model_type=lstm bound_output=True device=cuda solver=gurobi
# python scripts/train.py --config-name $1 lr=0.001 model_type=feedforward bound_output=True device=cuda loss_info.loss_fn=mse
# python scripts/train.py --config-name $1 lr=0.001 model_type=lstm bound_output=True device=cuda &
# python scripts/train.py --config-name $1 lr=0.0001 model_type=feedforward bound_output=True device=cuda
# python scripts/train.py --config-name $1 lr=0.001 model_type=rnn bound_output=True device=cuda &
# python scripts/train.py --config-name $1 lr=1 model_type=linear bound_output=True device=cuda

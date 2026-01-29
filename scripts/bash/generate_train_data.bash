CORES=$2
CONFIG=loco/cmippy/configs/$1.yaml
N_PER_PROBLEM=$3
SOLVER=$4

python loco/cmippy/gurobi/get_problem_representations.py --config_dir=$CONFIG
python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=$CORES --redo=1 --n_per_problem=$N_PER_PROBLEM --do_eval=1 --do_test=1 --do_train=1 --solver=$SOLVER
python loco/cmippy/gurobi/add_cols.py --config_dir=$CONFIG --solver=$SOLVER
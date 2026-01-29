
CORES=$2
CONFIG=loco/cmippy/configs/$1.yaml
N_PER_PROBLEM=$3

# python loco/cmippy/gurobi/get_problem_representations.py --config_dir=$CONFIG
python loco/cmippy/ecole/data.py --config_dir=$CONFIG --do_eval=0 --do_test=1 --do_train=0
python loco/cmippy/gurobi/add_cols.py --config_dir=$CONFIG --solver=ecole
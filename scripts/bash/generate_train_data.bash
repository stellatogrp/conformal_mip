CORES=$2
CONFIG=cmippy/configs/$1.yaml
SOLVER=$3

python cmippy/get_problem_representations.py --config_dir=$CONFIG
python cmippy/data.py --config_dir=$CONFIG --cores=$CORES --redo=1 --n_per_problem=1 --do_eval=1 --do_test=1 --do_train=1 --solver=$SOLVER
python cmippy/add_cols.py --config_dir=$CONFIG --solver=$SOLVER
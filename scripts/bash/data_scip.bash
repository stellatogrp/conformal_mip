
CONFIG=loco/cmippy/configs/$1.yaml

python loco/cmippy/gurobi/get_problem_representations.py --config_dir=$CONFIG

nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True &
nohup python loco/cmippy/gurobi/data.py --config_dir=$CONFIG --cores=1 --redo=0 --do_eval=0 --do_test=0 --do_train=1 --solver=scip --shuffle=True

python loco/cmippy/gurobi/add_cols.py --config_dir=$CONFIG --solver=scip
CORES=3
ALPHA=0.05
EPSILON=0.001
python scripts/test.py --experiment_name=OTS-medium --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=gurobi > logs/OTS-medium.log &
python scripts/test.py --experiment_name=MIS-medium --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=gurobi > logs/MIS-medium.log &
python scripts/test.py --experiment_name=MVC-medium --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=gurobi > logs/MVC-medium.log &
python scripts/test.py --experiment_name=indset --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=gurobi > logs/indset.log &
python scripts/test.py --experiment_name=GISP-easy --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=gurobi > logs/GISP-easy.log &
python scripts/test.py --experiment_name=MMCN-medium-BI --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=gurobi > logs/MMCN-medium-BI.log &
python scripts/test.py --experiment_name=NNV-easy --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=gurobi > logs/NNV-easy.log &
python scripts/test.py --experiment_name=CFLP-medium --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=gurobi > logs/CFLP-medium.log
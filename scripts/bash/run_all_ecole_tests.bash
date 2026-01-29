CORES=3
ALPHA=0.1
EPSILON=0.1
python scripts/test.py --experiment_name=NNV-easy --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=ecole
# python scripts/test.py --experiment_name=CFLP-medium --cores=$CORES --alpha=$ALPHA --epsilon=$EPSILON --solver=ecole

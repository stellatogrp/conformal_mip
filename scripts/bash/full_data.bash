SERIES=$1
TEST=$SERIES/test
TRAIN=$SERIES/train
VAL=$SERIES/val
bash scripts/bash/data.bash $TRAIN 10 5 & 
bash scripts/bash/data.bash $VAL 10 5 & 
bash scripts/bash/data.bash $TEST 10 1 & 
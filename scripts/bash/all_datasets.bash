conda activate loco
module load gurobi/12.0.0
bash scripts/bash/full_data.bash CA-very-easy &
bash scripts/bash/full_data.bash knapsack-easy &
bash scripts/bash/full_data.bash CFLP-easy &
bash scripts/bash/full_data.bash facloc-easy &
bash scripts/bash/full_data.bash SPRN-easy && fg
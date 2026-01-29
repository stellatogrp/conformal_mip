Code for the paper "Conformal Prediction for Early Stopping in Mixed Integer Optimization".
This code is anonymized and this branch was last modified on Jan 28th 2026.

INSTRUCTIONS:

1. Install necessary packages
```bash
pip install requirements.txt # required packages
pip install -e . # install cmippy package
```
Ensure you have either GUROBI or COPT installed with a valid lisence.

2. Download distributional MIPLIB instances from https://sites.google.com/usc.edu/distributional-miplib/home.
You should unzip the folder downloaded and place it in the 'instances' folder in this repository. 
There is a toy family included in this reposityro named 'CFLP_toy' which you can use as a template.
The instances we consider in the paper are:
    1. CFLP-medium
    2. GISP-easy
    3. MIS-medium
    4. MMCN-medium-BI
    5. MVC-medium
    6. OTS-medium
If you use one of these, all of the configs should be ready to go.

3. Adjust config file.
This repository uses mlflow to manage the experiments.
If in the step above you added a family of MIPs not from the above list, say called MIP-medium, you must create a config in cmippy/configs called MIP-medium.yaml and populate it with the appropriate settings (see one of the other configs for details).
Here is where you change the device the NN is stored on (default is CUDA), change size of neural network, change loss function, ect. Default settings are settings used in our paper.

from now on we will demonstrate on the toy instance, CFLP-toy, using the GUROBI solver.

3. Generate training data for the neural network.
```bash
bash scripts/bash/generate_train_data.bash CFLP-toy <N-CORES> gurobi
```
This will populate a folder in instance_data/gurobi/CFLP-toy with training data from the GUROBI solves.
This directory `optimizers` contains the code used to implement taKG, BO-EI, and Random Search, to evaluate the gains that stem from using an additional dimension. These were implemented based on the [BoTorch library](https://botorch.org/). On the other hand, HB was implemented based on the implementations found in public repositories of [BOHB](https://github.com/automl/HpBandSter) and [HyperJump](https://github.com/pedrogbmendes/HyperJump).

## How to reproduce the results in the paper:

To reproduce the results in the paper (comparison of different HPT strategies), you need to directly run the following scripts:

1) For all the takg-based baselines: `python3 takg.py`

2) For BO-EI: `python3 ei_advTrain.py`

3) For Random Search `python3 ei_advTrain.py` (you need to set in this file the variable `acqFunc=random`)

4) HyperBand: HB is based on the  [BOHB](https://github.com/automl/HpBandSter) and [HyperJump](https://github.com/pedrogbmendes/HyperJump), which contain and independent repository. To be able to run HB, you need to define the function to optimize (i.e., the model to train and the hyper-parameters).


You can select the model/dataset in the header of the file (argument called network). 
These scripts do not deploy the training in real-time (the training procedure data is read from a file saved on [dataset](https://github.com/pedrogbmendes/HPT_advTrain/tree/anonymize/datasets).
To run taKG, you also need to select the budgets/fidelities to use. You have 3 options:
```
budget_option = 0: both budgets - number of epochs and PGD iteration
budget_option = 1: number of epochs as budget
budget_option = 2: PGD iterations as budget
```
You can also set the bound to perform AT by setting it in the variable `Bound_epsilon`.


You need to install botorch, pytorch, and ConfigSpace libraries using pip3 (e.g., `pip3 install botorch`).

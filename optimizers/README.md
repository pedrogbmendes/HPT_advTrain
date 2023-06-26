This directory `optimizers` contains the code used to implement taKG, BO-EI, and Random Search. These were implemented based on the [BoTorch library](https://botorch.org/). On the other hand, HB was implemented based on the implementations found in public repositories of [BOHB](https://github.com/automl/HpBandSter) and [HyperJump](https://github.com/pedrogbmendes/HyperJump).


You can run these scripts but selecting the model/dataset in the header of the file (argument called network). 
These scripts do not deploy the training in real-time (the training procedure data is read from a file saved on [dataset](https://github.com/pedrogbmendes/HPT_advTrain/tree/main/datasets).
To run taKG, you also need to select the budgets/fidelities to use. You have 3 options:
```
budget_option = 0: both budgets - number of epochs and PGD iteration
budget_option = 1: number of epochs as budget
budget_option = 2: PGD iterations as budget
```
You can also set the bound to perform AT by setting it in the variable `Bound_epsilon`.


You need to install botorch, pytorch, and ConfigSpace libraries using pip3 (e.g., `pip3 install botorch`).

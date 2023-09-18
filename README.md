# Hyper-parameter Tuning for Adversarially Robust Models


This repository contains the scripts and data used and collected in our study entitled Hyper-parameter Tuning for Adversarially Robust Models. 

**Due to the anonymization of this repo, some of the links to directories of this repo might not work.**


##  [Train directory](https://github.com/pedrogbmendes/HPT_advTrain/tree/anonymize/train)

**The [directory `train`](https://github.com/pedrogbmendes/HPT_advTrain/tree/anonymize/train)  contains the scripts and information to deploy the training of models.**

To train the models, you can directly run the `python3 train.py` specifying the correct arguments:
```
  --type: type of training (standard training (AT) or adversarial training (AT)), choices=['std', 'robust']
  --alg: method to generate the perturbation when performing adversarial training, choices=['pgd', 'fgsm', 'pgd_rs', 'fgsm_rs', 'fgsm_free', 'fgsm_grad_align']
  --dataset: dataset/model to train, choices=['mnist', 'cifar10', 'imageNet', 'svhn']
  --ratio: percentage of resources for AT (%RAT), value between 0 and 1
  --ratio_adv: percentage of adversarial examples (%AE) when performing AT, , value between 0 and 1
  --epsilon: bound for the pertubation
  --num_iter: number of iterations for PGD
  --alpha: similar to the learning rate for PGD
  --stop: stop condition, choices=['epochs', 'time']
  --stop_val: the value of the stop condition
  --lr: learning rate of ST
  --momentum: the momentum of ST
  --batch: the batch size of ST
  --lr_adv: learning rate of AT
  --momentum_adv: momentum of AT
  --batch_adv: the batch size of AT
  --workers: number of workers
  --half_prec: half-precision computation
  --preTrain: whether the model should be saved/loaded, choices=[0,1,2] (0 trains from the beginning, 1 trains from the beginning and saves it (used just for ST), 2 loads the ST model if exists and performs AT)
```

The benchmarks should be in a directory called `../data`.
To parallelize and automatize the deployment and speed up the training, you can also run `python3 run.py`. In this file, you can directly specify all the arguments to run the train.py file, and the script will deploy in a sequential (1 worker) or parallel way (several workers) the training process.
Do not forget to install pytorch.
These scripts were used to train a given model using a specific benchmark using different HPs configurations (see Section 3.2 of our paper and Figures 1, 2, and 3).




## [Optimizers directory](https://github.com/pedrogbmendes/HPT_advTrain/tree/anonymize/optimizers)  

**The [directory `optimizers`](https://github.com/pedrogbmendes/HPT_advTrain/tree/anonymize/optimizers) contains the code to implement the hyper-parameter tuning optimizers** compared in this work (namely, taKG, HB, BO-EI, and Random Search).

This directory `optimizers` contains the code used to implement taKG, BO-EI, and Random Search, to evaluate the gains that stem from using an additional dimension. These were implemented based on the [BoTorch library](https://botorch.org/). On the other hand, HB was implemented based on the implementations found in public repositories of [BOHB](https://github.com/automl/HpBandSter) and [HyperJump](https://github.com/pedrogbmendes/HyperJump).

### How to reproduce the results in the paper:

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



## [Datasets directory](https://github.com/pedrogbmendes/HPT_advTrain/tree/anonymize/datasets)  

**The [directory `datasets`](https://github.com/pedrogbmendes/HPT_advTrain/tree/anonymize/datasets)  contains the data collected in our study.** We are making this data publicly accessible in the hope that it will aid the design of future HPT methods specialized for Adversarial Training. 
This directory contains the data collected for our study entitled **Hyper-parameter Tuning for Adversarially Robust Models**.
The files can also be downloaded from this [link](https://drive.google.com/drive/folders/1qV_fiJA_JzEin-SscksE0tlC5Adt187X?usp=sharing). 
(The file containing the data of CNN Cifar10 was too big to be uploaded to this repository, so you need to download it from the link above).

Each CSV file contains the following fields:
```
  alg: method to generate the perturbation when performing adversarial training or standard training
  epochs: epoch number
  lr: learning rate of ST
  momentum: the momentum of ST
  batch: the batch size of ST
  lrAdv: learning rate of AT
  momentumAdv: momentum of AT
  batchAdv: the batch size of AT
  ratio: percentage of resources for AT (%RAT)
  ratioAdv: percentage of adversarial examples (%AE) when performing AT
  epsilon: bound for the pertubation
  numIt: number of iterations for PGD
  alpha: similar to the learning rate for PGD
  epsilonTest: bound  used to generate the perturbation to test the model
  numItTest: number of iterations for PGD  used to generate the perturbation to test the model
  alphaTest: similar to the learning rate for PGD used to generate the perturbation to test the model 
  trainingError: training error
  testingError: standard test error
  advError: adversarial test error
  trainingTime: training time
  testingTime: standard testing time
  adversarialTestingTime: adversarial testing time
```




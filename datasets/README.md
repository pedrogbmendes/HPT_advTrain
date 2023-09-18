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


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

The datasets should be in a directory called `../data`.
To parallelize and automatize the deployment and speed up the training, you can also run   `python3 run.py`. In this file, you can directly specify all the arguments to run the train.py file, and the script will deploy in a sequential (1 worker) or parallel way (several workers) the training process.




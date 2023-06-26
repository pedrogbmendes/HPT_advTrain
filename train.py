import os
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from torch.nn import init

#models
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.datasets.vision import VisionDataset

#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.data.distributed import DistributedSampler
#import torch.multiprocessing as mp

import argparse

import pickle, time, random
from PIL import Image

from math import log10, sqrt
import numpy as np

from models import *

imageNet_original = False

torch.manual_seed(0)

random.seed(10000)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict



class ImageNetLoader(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        #print(self.data)
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            #print("tensor")

        x = self.data['x'][idx] 
        y = self.data['y'][idx] 

        if self.transform:
            x = self.transform(x)

        return x, y
        
    def __len__(self):
        #return self.data['x'].shape[0]
        return len(self.data['x'])


class SmallImagenet(VisionDataset):
    # code taken from https://github.com/landskape-ai/ImageNet-Downsampled
    train_list = ['train_data_batch_{}'.format(i + 1) for i in range(10)]
    val_list = ['val_data']

    def __init__(self, root="data", size=32, train=True, transform=None, classes=None):
        super().__init__(root, transform=transform)
        file_list = self.train_list if train else self.val_list
        self.data = []
        self.targets = []
        for filename in file_list:
            filename = os.path.join(self.root, filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f)
            self.data.append(entry['data'].reshape(-1, 3, size, size))
            self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.concatenate(self.targets).astype(int) - 1

        if classes is not None:
            classes = np.array(classes)
            filtered_data = []
            filtered_targets = []

            for l in classes:
                idxs = self.targets == l
                filtered_data.append(self.data[idxs])
                filtered_targets.append(self.targets[idxs])

            self.data = np.vstack(filtered_data)
            self.targets = np.concatenate(filtered_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class dataset():

    def __init__(self, dataset_name="mnist", batch_size = 100,  batch_size_adv = 100, ratio=1.0):
        batch_size_test = 256
        
        if dataset_name == "mnist":
            mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
            mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())

            _sizeAdv = int(len(mnist_train)*ratio)
            _size = int(len(mnist_train) - _sizeAdv)

            train_set, trainAdv_set = torch.utils.data.random_split(mnist_train, [_size, _sizeAdv])

            if _size == 0 or batch_size==0 or ratio==1.0:
                self.train_loader = None
            else:
                self.train_loader = DataLoader(train_set,    batch_size = batch_size, shuffle=True)

            if _sizeAdv == 0 or batch_size_adv==0 or ratio==0.0:
                self.trainAvd_loader = None
            else:
                self.trainAvd_loader = DataLoader(trainAdv_set, batch_size = batch_size_adv, shuffle=True)

            self.test_loader = DataLoader(mnist_test, batch_size = batch_size_test, shuffle=False)


        elif dataset_name ==  "cifar10":
            cifar10_mean = (0.4914, 0.4822, 0.4465)
            cifar10_std = (0.2471, 0.2435, 0.2616)

            mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
            std = torch.tensor(cifar10_std).view(3,1,1).cuda()

            upper_limit = ((1 - mu)/ std)
            lower_limit = ((0 - mu)/ std)

            #pre-processing 
            train_transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(cifar10_mean, cifar10_std), ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),])

            cifar10_train = datasets.CIFAR10("../data", train=True, download=True, transform=train_transform)
            cifar10_test = datasets.CIFAR10("../data", train=False, download=True, transform=test_transform)

            #cifar10_train = datasets.CIFAR10("../data", train=True, download=True, transform=transforms.ToTensor())
            #cifar10_test = datasets.CIFAR10("../data", train=False, download=True, transform=transforms.ToTensor())

            _sizeAdv = int(len(cifar10_train)*ratio)
            _size = int(len(cifar10_train) - _sizeAdv)
            #self.train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle=True)

            train_set, trainAdv_set = torch.utils.data.random_split(cifar10_train, [_size, _sizeAdv])

            #self.train_loader    = DataLoader(train_set , batch_size = batch_size, shuffle=True)
            #self.trainAvd_loader = DataLoader(trainAdv_set, batch_size = batch_size_adv, shuffle=True)

            if _size == 0 or batch_size==0 or ratio==1.0:
                self.train_loader = None
            else:
                self.train_loader = DataLoader(train_set,    batch_size = batch_size, shuffle=True)

            if _sizeAdv == 0 or batch_size_adv==0 or ratio==0.0:
                self.trainAvd_loader = None
            else:
                self.trainAvd_loader = DataLoader(trainAdv_set, batch_size = batch_size_adv, shuffle=True)

            self.test_loader = DataLoader(cifar10_test, batch_size = batch_size_test, shuffle=False)


        elif dataset_name ==  "imageNet":
            workers = 5

            if imageNet_original:
                print("Original dataset")
                traindir = '../data/imageNet/train'                
                valdir = '../data/imageNet/val'                
                crop_size = 224


                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                _train_set = datasets.ImageFolder( traindir, transforms.Compose([
                        transforms.RandomResizedCrop(crop_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),normalize,
                    ]))

                _test_set = datasets.ImageFolder(valdir, transforms.Compose([
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(),normalize,
                    ]))

            else:
                print("Downsampled dataset")

                root_dir = '../data/imageNet/'

                resolution=32 
                classes=1000
                
                normalize = transforms.Normalize(mean=[0.4810,0.4574,0.4078], std=[0.2146,0.2104,0.2138])

                tf_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
                tf_test = transforms.Compose([transforms.ToTensor(),normalize,])

                _train_set = SmallImagenet(root=root_dir, size=resolution, train=True, transform=tf_train, classes=range(classes)) 
                _test_set = SmallImagenet(root=root_dir, size=resolution, train=False, transform=tf_test, classes=range(classes)) 

                #OLD VERSION                         
                # _x_train = None
                # _y_train = None
                # print("Reading dataset to train")

                # for i in range(1, 11): 
                #     _x, _y = self.load_data(i)

                #     if i == 1:
                #         _x_train =_x
                #         _y_train =_y
                #     else:
                #         _x_train = np.concatenate((_x_train,_x))
                #         _y_train = np.concatenate((_y_train,_y))

                # print("Complete. \n Reading dataset to test")
                # _x_test, _y_test = self.load_data(-1, train=False)
                # print("Complete.")
                # x_train, y_train = _x_train, _y_train
                # x_test, y_test = _x_test, _y_test

                # #for curr_index in range(len(_x_train)):
                # #    x_train  = Image.fromarray(_x_train[curr_index])

                # data_train = dict(x=x_train, y=y_train)
                # data_test = dict(x=x_test, y=y_test)

                # #torchvision.datasets.ImageNet(root: str, split: str = 'train', **kwargs: Any)
                # tf=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

                # _train_set = ImageNetLoader(data_train, transform=tf)
                # _test_set = ImageNetLoader(data_test, transform=tf)

            _sizeAdv = int(len(_train_set)*ratio)
            _size = int(len(_train_set) - _sizeAdv)

            train_set, trainAdv_set = torch.utils.data.random_split(_train_set, [_size, _sizeAdv])

            if _size == 0 or batch_size==0 or ratio==1.0:
                self.train_loader = None
            else:
                self.train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=workers, pin_memory=True,)

            if _sizeAdv == 0 or batch_size_adv==0 or ratio==0.0:
                self.trainAvd_loader = None
            else:
                self.trainAvd_loader = DataLoader(trainAdv_set, batch_size = batch_size_adv, shuffle=True, num_workers=workers, pin_memory=True,)

            #self.test_loader = DataLoader(imageNet_test, batch_size = batch_size, shuffle=False)
            self.test_loader = DataLoader(_test_set, batch_size = batch_size_test, shuffle=False, num_workers=workers, pin_memory=True,)

        elif dataset_name ==  "svhn":
            #pre-processing 
            #transform = transforms.Compose([transforms.RandomCrop(32, padding=4)] + [transforms.ToTensor()])

            svhn_train = datasets.SVHN("../data", split='train', download=True, transform=transforms.ToTensor())
            svhn_test = datasets.SVHN("../data", split='test', download=True, transform=transforms.ToTensor())
            #self.train_loader = DataLoader(svhn_train, batch_size = batch_size, shuffle=True)

            _sizeAdv = int(len(svhn_train)*ratio)
            _size = int(len(svhn_train) - _sizeAdv)

            train_set, trainAdv_set = torch.utils.data.random_split(svhn_train, [_size, _sizeAdv])

            #self.train_loader    = DataLoader(train_set , batch_size = batch_size, shuffle=True)
            #self.trainAvd_loader = DataLoader(trainAdv_set , batch_size = batch_size_adv, shuffle=True)

            if _size == 0 or batch_size==0 or ratio==1.0:
                self.train_loader = None
            else:
                self.train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, pin_memory=True,)

            if _sizeAdv == 0 or batch_size_adv==0 or ratio==0.0:
                self.trainAvd_loader = None
            else:
                self.trainAvd_loader = DataLoader(trainAdv_set, batch_size = batch_size_adv, shuffle=True, pin_memory=True,)


            self.test_loader = DataLoader(svhn_test, batch_size = batch_size_test, shuffle=False)

        else:
            raise("dataset not implementex")
        

    def load_data(self, idx, train=True, dir='../data'):
        if train:
            input_file = '/imageNet/train_data_batch_'
            #input_file = '/imageNet/Imagenet32_train/train_data_batch_'
            d = unpickle(dir+input_file+str(idx))
        else:
            input_file = '/imageNet/val_data'
            d = unpickle(dir+input_file)

        x = d['data']
        y = d['labels']

        y = [i-1 for i in y]

        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3))
        return x, y


class trainModel():
    def __init__(self, device, half_prec=False):
        self.device = device
        self.half_prec=half_prec

        if self.half_prec:
            # we only used mixed precision in imageNet
            #self.model, self.opt = amp.initialize(self.model, self.opt, opt_level="O1")
            # Creates once at the beginning of training
            self.scaler = torch.cuda.amp.GradScaler()

        return


    def writeResult(self, filename, data):
        trainTime, train_err, train_loss = data[0]
        testTime, test_err, test_loss = data[1]
        advTestTime_pgd, adv_err_pgd, adv_loss_pgd = data[2]

        str1 = "total_time=" + str(trainTime) + ";"
        str1 += "\ntraining_error=" + str(train_err) + ";"
        str1 += "\ntraining_loss=" + str(train_loss) + ";"

        str1 += "\ntesting_time=" + str(testTime) + ";"
        str1 += "\ntest_error=" + str(test_err) + ";"
        str1 += "\ntest_loss=" + str(test_loss) + ";"

        str1 += "\navd_test_time_pgd=" + str(advTestTime_pgd) + ";"
        str1 += "\nadversarial_error_pgd=" + str(adv_err_pgd) + ";"
        str1 += "\nadversarial_loss_pgd=" + str(adv_loss_pgd) + ";"

        if len(data) == 4:
            advTestTime_fgsm, adv_err_fgsm, adv_loss_fgsm = data[3]
            str1 += "\navd_test_time_fgsm=" + str(advTestTime_fgsm) + ";"
            str1 += "\nadversarial_error_fgsm=" + str(adv_err_fgsm) + ";"
            str1 += "\nadversarial_loss_fgsm=" + str(adv_loss_fgsm) + ";"

        f = open(filename, "w")
        f.write(str1)
        f.close()


    def saveModel(self, saveModel, state, filename, ratio=1.0):
        if not saveModel: return

        #torch.save(model.state_dict(), name)
        torch.save(state,  "models/" + filename + "_ratio" + str(ratio) + ".pt")
        

    def LoadModel(self, model, opt, filename):
        filename = "models/" + filename + ".pt"
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            training_time = checkpoint['training_time']
            train_err = checkpoint['error']
            train_loss = checkpoint['loss']

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']+1))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            model, opt, training_time, train_err, train_loss, start_epoch = None, None, None, None, None, None

        return model, opt, training_time, train_err, train_loss, start_epoch


    def updateLogs(self, oldName, NewName, alg, ratio ,epsilon, numIt, alpha, ratioADV):

        with open("./logs1/logs_" + oldName + '.txt', 'r') as f_read:
            with open("./logs/logs_" + NewName + '.txt', 'w') as f_write:
                for i, line in enumerate(f_read): 
                    if i == 0 :
                        f_write.write(line)
                    else:
                        #it,alg,ratio,epsilon,numIt,alpha,ratioAdv,algTest,epsilonTest,numItTest,alphaTest,Error,Loss,testingTime,trainTime
                        aux = line.split(",")
                        iteration = aux[0]
                        algTest = aux[7]
                        eps_test = aux[8]
                        num_iterTest = aux[9]
                        alpha_test = aux[10]
                        adv_err = aux[11]
                        adv_loss = aux[12]
                        advTestTime = aux[13]
                        trainTime = aux[14]


                        str_write = str(iteration) + "," + alg + "," + str(ratio) + "," + str(epsilon) + "," + str(numIt) + "," + str(alpha) + "," + str(ratioADV) + \
                                        "," + algTest +"," + str(eps_test) + "," + str(num_iterTest) + "," + str(alpha_test) + "," + \
                                        str(adv_err) + "," + str(adv_loss) + "," + str(advTestTime) + "," + str(trainTime) #+ "\n"          
                        f_write.write(str_write)
 

    def standard_train(self, model, modelName, loader, dataset, opt, iterations=10, saveModel=False):
    #def standard_train(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, iterations=10, saveModel=False):
        '''training a standard model.''' 
        t1 = time.time()
        print("Standard training")

        for t in range(iterations):
            train_err, train_loss = self.epoch(loader.train_loader, model, opt)

            self.testModel_logs(dataset, modelName, t, 'standard', 0 ,0, 0, 0, 0, time.time() - t1)

        trainTime = time.time() - t1
        
        #self.saveModel(saveModel, model, modelName)
        
        return (trainTime, train_err, train_loss)


    def standard_train_save(self, model, modelName, loader, dataset, opt, iterations=10, stop=False):
        '''training a standard model with checkpoint and saving the model.''' 
        
        list_ratios = []
        if not os.path.isfile("models/" + modelName + "_ratio0.3.pt") or not os.path.isfile("logs1/logs_" + modelName + "_ratio0.3.txt"):
            list_ratios.append(int(iterations*0.3)) 
        if not os.path.isfile("models/" + modelName + "_ratio0.5.pt") or not os.path.isfile("logs1/logs_" + modelName + "_ratio0.5.txt"):
            list_ratios.append(int(iterations*0.5)) 
        if not os.path.isfile("models/" + modelName + "_ratio0.7.pt") or not os.path.isfile("logs1/logs_" + modelName + "_ratio0.7.txt"):
            list_ratios.append(int(iterations*0.7)) 

        if stop and len(list_ratios) == 0:
            print("models trained")
            return -1, -1, -1

        maxIt = max(list_ratios) if stop else iterations
            
        print("Standard training")
        t1 = time.time()
        for counter in range(maxIt):
            #print("epoch number " + str(counter))
            train_err, train_loss = self.epoch(loader.train_loader, model, opt)

            list_ratios_2_test = []
            if int(iterations*0.3) in list_ratios and counter < int(iterations*0.3):
                #self.testModel_logs(dataset, modelName, counter, 'standard', 0.3 ,0, 0, 0, 0, time.time() - t1, check=True) #logs of checkpointing
                list_ratios_2_test.append(0.3)
            if int(iterations*0.5) in list_ratios and counter < int(iterations*0.5):
                #self.testModel_logs(dataset, modelName, counter, 'standard', 0.5 ,0, 0, 0, 0, time.time() - t1, check=True)
                list_ratios_2_test.append(0.5)
            if int(iterations*0.7) in list_ratios and counter < int(iterations*0.7):
                #self.testModel_logs(dataset, modelName, counter, 'standard', 0.7 ,0, 0, 0, 0, time.time() - t1, check=True)
                list_ratios_2_test.append(0.7)
            if not stop:
                list_ratios_2_test.append(1.0)

            if list_ratios_2_test: self.testModel_logs(dataset, modelName, counter, 'standard', list_ratios_2_test, 0, 0, 0, 0, time.time() - t1)


            #if not stop: self.testModel_logs(dataset, modelName, counter, 'standard', 0 ,0, 0, 0, 0, time.time() - t1, check=False) #normal logs


            if counter+1 in list_ratios:
                if int(iterations*0.3) == counter+1: _ratioStd = 0.3
                elif int(iterations*0.5) == counter+1:  _ratioStd = 0.5
                else: _ratioStd = 0.7 #elif int(iterations*0.7) == t+1:
                
                print("saving model on epoch " + str(counter) + " with standard ratio " + str(_ratioStd))

                self.saveModel(True, {
                    'epoch': counter,
                    'state_dict': model.state_dict(),
                    'training_time': time.time() - t1,
                    'error': train_err,
                    'loss': train_loss,
                    'optimizer' : opt.state_dict(),}, modelName, _ratioStd)

        trainTime = time.time() - t1

        if stop: return (-1, -1, -1)
        
        return (trainTime, train_err, train_loss)


    def standard_train_time(self, model, modelName, loader, dataset, opt, stop_time=100, saveModel=False):
        '''training a standard model stop condition time.'''
        
        start_time = time.time()
        t = 0
        while time.time() - start_time < stop_time:
            train_err, train_loss = self.epoch(loader.train_loader, model, opt)

            #evaluate the model after each epoch
            self.testModel_logs(dataset, modelName, t, 'standard', 0 ,0, 0, 0, 0, time.time() - start_time)
            t += 1
            
        trainTime = time.time() - start_time

        return (trainTime, train_err, train_loss)

                
    def standard_pgd_train(self, model, modelName, loader, dataset, opt, iterations=10, saveModel=False,ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        
        t1 = time.time()
        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        counter = 0 
        for t in range(int(iterations*(1-ratio))):

            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for t in range(int(iterations*ratio)):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1

        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_pgd_train_load(self, model, modelName, loader, dataset, opt, iterations=10, ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        
        #load the model
        #logs_modelimageNet_std_train_lr0.1_momentum0.9_batch512_lrAdv0.0_momentumAdv0.0_batchAdv0_ratio0.3
        #logs_modelimageNet_robust_PGD_eps4.0_numIt20_alpha0.01_ratio1.0_ratioadv1.0_lr0.1_momentum0.9_batch512_lrAdv0.1_momentumAdv0.9_batchAdv512
        trainSTD = True
        if ratio > 0.0 and ratio < 1.0:
            auxName = modelName.split("_")
            ratioStd = 1 - ratio

            old_name = auxName[0] + "_std_train_" + auxName[8] + "_" + auxName[9] + "_" + auxName[10]
            old_name += "_lrAdv0.0_momentumAdv0.0_batchAdv0_ratio" + str(ratioStd)[0:3]
            
            _model, _opt, trainTime, train_err1, train_loss1, counter = self.LoadModel(model, opt, old_name)

            if _model is not None:
                self.updateLogs(old_name, modelName, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv)
                t1 = time.time()-trainTime
                model = _model
                opt = _opt
                counter += 1
                trainSTD = False


        if trainSTD:    
            t1 = time.time()
            train_err1, train_loss1 = 0,0
            train_err2, train_loss2 = 0,0

            counter = 0 
            for _ in range(int(iterations*(1-ratio))):
                print("epoch number " + str(counter))
                train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
                self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1)
                counter += 1


        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for t in range(int(iterations*ratio)):
            print("epoch number " + str(counter))

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1

        #self.saveModel(saveModel, model, modelName)
        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_pgd_train_time(self, model, modelName, loader, dataset, opt, stop_time=100, saveModel=False, ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()

        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        start_time = time.time()
        #t = 0
        counter = 0 

        while time.time() - start_time < stop_time*(1-ratio):

            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv
            
        start_time = time.time()
        #t = 0
        while time.time() - start_time < stop_time*(ratio):
            
            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_pgd', ratio, eps_train, num_iterTrain, alpha_train, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1

        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsm_train(self, model, modelName, loader, dataset, opt, iterations=10, saveModel=False,ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()
        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        counter = 0 
        for t in range(int(iterations*(1-ratio))):

            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for t in range(int(iterations*ratio)):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1
        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsm_train_load(self, model, modelName, loader, dataset, opt, iterations=10, ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        trainSTD = True
        if ratio > 0.0 and ratio < 1.0:
            #load the model
            #logs_modelimageNet_std_train_lr0.1_momentum0.9_batch512_lrAdv0.0_momentumAdv0.0_batchAdv0_ratio0.3
            #logs_modelimageNet_robust_FGSM_eps2.0_ratio0.3_ratioadv0.3_lr0.1_momentum0.9_batch512_lrAdv0.1_momentumAdv0.9_batchAdv512.txt
            auxName = modelName.split("_")
            ratioStd = 1.0 - ratio

            old_name = auxName[0] + "_std_train_" + auxName[6] + "_" + auxName[7] + "_" + auxName[8]
            old_name += "_lrAdv0.0_momentumAdv0.0_batchAdv0_ratio" + str(ratioStd)[0:3]
            
            _model, _opt, trainTime, train_err1, train_loss1, counter = self.LoadModel(model, opt, old_name)

            if _model is not None:
                self.updateLogs(old_name, modelName, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv)
                t1 = time.time()-trainTime
                opt = _opt
                model = _model
                counter += 1
                trainSTD = False


        if trainSTD:
            t1 = time.time()
            train_err1, train_loss1 = 0,0
            train_err2, train_loss2 = 0,0

            counter = 0 
            for t in range(int(iterations*(1-ratio))):
                print("epoch number " + str(counter))
                train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
                self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
                counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for _ in range(int(iterations*ratio)):
            print("epoch number " + str(counter))
            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsm_train_time(self, model, modelName, loader, dataset, opt, nesterov=False, stop_time=100, saveModel=False,ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()

        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        start_time = time.time()
        counter = 0
        while time.time() - start_time < stop_time*(1-ratio):
           
            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        start_time = time.time()
        # t = 0
        while time.time() - start_time < stop_time*(ratio):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1
        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsmRs_train(self, model, modelName, loader, dataset, opt, iterations=10, saveModel=False,ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()
        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        counter = 0 
        for t in range(int(iterations*(1-ratio))):

            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_rs', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for t in range(int(iterations*ratio)):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_rs", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_rs', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1
        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsmRs_train_time(self, model, modelName, loader, dataset, opt, nesterov=False, stop_time=100, saveModel=False,ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()

        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        start_time = time.time()
        counter = 0
        while time.time() - start_time < stop_time*(1-ratio):
           
            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_rs', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        start_time = time.time()
        # t = 0
        while time.time() - start_time < stop_time*(ratio):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_rs", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_rs', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1
        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsmFree_train(self, model, modelName, loader, dataset, opt, iterations=10, saveModel=False,ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()
        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        counter = 0 
        for t in range(int(iterations*(1-ratio))):

            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_free', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for t in range(int(iterations*ratio)):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_free", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_free', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1
        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsmFree_train_time(self, model, modelName, loader, dataset, opt, nesterov=False, stop_time=100, saveModel=False,ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()

        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        start_time = time.time()
        counter = 0
        while time.time() - start_time < stop_time*(1-ratio):
           
            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_free', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        start_time = time.time()
        # t = 0
        while time.time() - start_time < stop_time*(ratio):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_free", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_free', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1
        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsmGrad_align_train(self, model, modelName, loader, dataset, opt, iterations=10, saveModel=False,ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()
        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        counter = 0 
        for t in range(int(iterations*(1-ratio))):

            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_grad_align', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        for t in range(int(iterations*ratio)):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_grad_align", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_grad_align', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1
        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def standard_fgsmGrad_align_train_time(self, model, modelName, loader, dataset, opt, nesterov=False, stop_time=100, saveModel=False,ratio=1,  eps_train=0.1, ratio_adv=1, lr_adv=1e-1, momentum_adv=0):
        '''training a standard model and then make it adversarial.'''
        t1 = time.time()

        train_err1, train_loss1 = 0,0
        train_err2, train_loss2 = 0,0

        start_time = time.time()
        counter = 0
        while time.time() - start_time < stop_time*(1-ratio):
           
            train_err1, train_loss1 = self.epoch(loader.train_loader, model, opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_grad_align', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        #update hyper-parameter for adversarial training
        for param_group in opt.param_groups:
            param_group["lr"] = lr_adv
            param_group["momentum"] = momentum_adv

        start_time = time.time()
        # t = 0
        while time.time() - start_time < stop_time*(ratio):

            train_err2, train_loss2 = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_grad_align", dataset, epsilon=eps_train, ratio=ratio_adv, opt=opt)
            self.testModel_logs(dataset, modelName, counter, 'std_fgsm_grad_align', ratio, eps_train, 0, 0, ratio_adv, time.time() - t1)
            counter += 1

        trainTime = time.time() - t1
        #self.saveModel(saveModel, model, modelName)

        train_err = (train_err1 + train_err2) / 2.0
        train_loss = (train_loss1 + train_loss2) / 2.0

        return (trainTime, train_err, train_loss)


    def printError():
        print("ERROR: FUNCTION NOT IN USE. PLEASE USE OTHER FUNCTION TO TRAIN.")


    def adversarial_train_pgd(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, iterations=10, 
                                saveModel=False, ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01):
        '''training a adversatial model using pgd_linf '''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        for t in range(iterations):

            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'pgd', 1, eps_train, num_iterTrain, alpha_train, ratio, time.time() - t1)

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_PGD_eps" + str(eps_train) + "_numIt" + str(num_iterTrain) + "_alpha" + str(alpha_train) +  "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")
        
        return (trainTime, train_err, train_loss)


    def adversarial_train_pgd_time(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, stop_time=100, 
                                saveModel=False, ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01):
        '''training a adversatial model using pgd_linf '''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        start_time = time.time()
        t = 0
        while time.time() - start_time < stop_time:
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd", dataset, epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'pgd', 1, eps_train, num_iterTrain, alpha_train, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2
            t += 1

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_PGD_eps" + str(eps_train) + "_numIt" + str(num_iterTrain) + "_alpha" + str(alpha_train) +  "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")
        
        return (trainTime, train_err, train_loss)


    def adversarial_train_pgd_rs(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, iterations=10, 
                                saveModel=False, ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01):
        '''training a adversatial model using pgd_linf '''
        self.printError()

        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        for t in range(iterations):
      
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd_rs", dataset, epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'pgd_rs', 1, eps_train, num_iterTrain, alpha_train, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_PGDrs_eps" + str(eps_train) + "_numIt" + str(num_iterTrain) + "_alpha" + str(alpha_train) +  "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")
        
        return (trainTime, train_err, train_loss)


    def adversarial_train_pgd_rs_time(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, stop_time=100, 
                                saveModel=False, ratio=1, num_iterTrain=20, eps_train=0.1, alpha_train=0.01):
        '''training a adversatial model using pgd_linf '''
        self.printError()

        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")
        start_time = time.time()
        t = 0
        while time.time() - start_time < stop_time:
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "pgd_rs", dataset, epsilon=eps_train, num_iter=num_iterTrain, alpha=alpha_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'pgd_rs', 1, eps_train, num_iterTrain, alpha_train, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2
            t += 1

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_PGDrs_eps" + str(eps_train) + "_numIt" + str(num_iterTrain) + "_alpha" + str(alpha_train) +  "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")
        
        return (trainTime, train_err, train_loss)


    def adversarial_train_fgsm(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, iterations=10, 
                                saveModel=False, ratio=1, eps_train=0.1):
        '''training a adversatial model using fgsm '''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #opt = optim.SGD(model.parameters(), lr=1e-1)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        for t in range(iterations):
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'fgsm', 1, eps_train, 0, 0, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_FGSM_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")

        return (trainTime, train_err, train_loss)


    def adversarial_train_fgsm_time(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, stop_time=100, 
                                saveModel=False, ratio=1, eps_train=0.1):
        '''training a adversatial model using fgsm '''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #opt = optim.SGD(model.parameters(), lr=1e-1)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        start_time = time.time()
        t = 0
        while time.time() - start_time < stop_time:
           
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm", dataset, epsilon=eps_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'fgsm', 1, eps_train, 0, 0, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2
            t += 1

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_FGSM_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")

        return (trainTime, train_err, train_loss)


    def adversarial_train_fgsm_rs(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, iterations=10, 
                                saveModel=False, ratio=1, eps_train=0.1):
        '''training a adversatial model using fgsm '''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        for t in range(iterations):
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_rs", dataset, epsilon=eps_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'fgsm_rs', 1, eps_train, 0, 0, ratio, time.time() - t1)

            # if t == 4:
                # for param_group in opt.param_groups:
                    # param_group["lr"] = 1e-2

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_FGSMrs_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")

        return (trainTime, train_err, train_loss)


    def adversarial_train_fgsm_rs_time(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, stop_time=100, 
                                saveModel=False, ratio=1, eps_train=0.1):
        '''training a adversatial model using fgsm '''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        start_time = time.time()
        t = 0
        while time.time() - start_time < stop_time:
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_rs", dataset, epsilon=eps_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'fgsm_rs', 1, eps_train, 0, 0, ratio, time.time() - t1)

            # if t == 4:
                # for param_group in opt.param_groups:
                    # param_group["lr"] = 1e-2
            t += 1

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_FGSMrs_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")

        return (trainTime, train_err, train_loss)


    def adversarial_train_fgsm_free(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, iterations=10, 
                                saveModel=False, ratio=1, eps_train=0.1):
        '''training a adversatial model using fgsm free'''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        for t in range(iterations):
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_free", dataset, epsilon=eps_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'fgsm_free', 1, eps_train, 0, 0, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_FGSMfree_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")

        return (trainTime, train_err, train_loss)


    def adversarial_train_fgsm_free_time(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, stop_time=100, 
                                saveModel=False, ratio=1, eps_train=0.1):
        '''training a adversatial model using fgsm free'''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        start_time = time.time()
        t = 0
        while time.time() - start_time < stop_time:
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_free", dataset, epsilon=eps_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'fgsm_free', 1, eps_train, 0, 0, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2
            t += 1

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_FGSMfree_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")

        return (trainTime, train_err, train_loss)


    def adversarial_train_fgsm_grad_align(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, iterations=10, 
                                saveModel=False, ratio=1, eps_train=0.1):
        '''training a adversatial model using fgsm with gradient alignment '''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #opt = optim.SGD(model.parameters(), lr=1e-1)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        for t in range(iterations):
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_grad_align", dataset, epsilon=eps_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'fgsm_grad_align', 1, eps_train, 0, 0, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_FGSMgradAlign_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")

        return (trainTime, train_err, train_loss)


    def adversarial_train_fgsm_grad_align_time(self, model, modelName, loader, dataset, lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False, stop_time=100,
                                saveModel=False, ratio=1, eps_train=0.1):
        '''training a adversatial model using fgsm '''
        self.printError()
        #training the model
        t1 = time.time()
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        #if self.half_prec:
        #    model, opt = amp.initialize(model, opt, opt_level="O1")

        start_time = time.time()
        t = 0
        while time.time() - start_time < stop_time:
            
            train_err, train_loss = self.epoch_adversarial(loader.trainAvd_loader, model, "fgsm_grad_align", dataset, epsilon=eps_train, ratio=ratio, opt=opt)
            self.testModel_logs(dataset, modelName, t, 'fgsm_grad_align', 1, eps_train, 0, 0, ratio, time.time() - t1)

            # if t == 4:
            #     for param_group in opt.param_groups:
            #         param_group["lr"] = 1e-2
            t += 1

        trainTime = time.time() - t1

        #if saveModel: torch.save(model.state_dict(), "models/model" + dataset + "_robust_FGSMgradAlign_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt")
        if saveModel: torch.save(model.state_dict(), "models/" + modelName + ".pt")

        return (trainTime, train_err, train_loss)


    def standard_test(self, model, loader):
        '''evaluating standard error.'''
        t3 = time.time()
        test_err, test_loss = self.epoch(loader.test_loader, model)
        testTime = time.time() - t3

        return (testTime, test_err, test_loss)


    def adversarial_test_pgd(self, model, loader, num_iterTest=20, eps_test=0.1, alpha_test=0.01):
        '''evaluating adversarial error using pgd.'''
        #not in use
        t4 = time.time()
        adv_err_pgd, adv_loss_pgd = self.epoch_adversarial(loader.test_loader, model, "pgd", "", epsilon=eps_test, num_iter=num_iterTest, alpha=alpha_test)
        advTestTime_pgd = time.time() - t4

        return (advTestTime_pgd, adv_err_pgd, adv_loss_pgd)


    def adversarial_test_pgd_rs(self, model, loader, num_iterTest=20, eps_test=0.1, alpha_test=0.01):
        '''evaluating adversarial error using pgd.'''
        #not in use
        t4 = time.time()
        adv_err_pgd, adv_loss_pgd = self.epoch_adversarial(loader.test_loader, model, "pgd_rs", "", epsilon=eps_test, num_iter=num_iterTest, alpha=alpha_test)
        advTestTime_pgd = time.time() - t4

        return (advTestTime_pgd, adv_err_pgd, adv_loss_pgd)


    def adversarial_test_fgsm_rs(self, model, loader, eps_test=0.1):
        '''evaluating adversarial error using fgsm.'''
        #not in use
        t4 = time.time()
        adv_err_fgsm, adv_loss_fgsm = self.epoch_adversarial(loader.test_loader, model, "fgsm_rs", "", epsilon=eps_test)
        advTestTime_fgsm = time.time() - t4

        return (advTestTime_fgsm, adv_err_fgsm, adv_loss_fgsm)


    def epoch(self, loader, model, opt=None):
        """Standard training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.

        for X,y in loader:
            X,y = X.to(self.device), y.to(self.device) # len of bacth size
            if self.half_prec: 
                # Runs the forward pass with autocasting.
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    yp = model(X)
                    loss = nn.CrossEntropyLoss().cuda()(yp,y)
                
                if opt:  
                    opt.zero_grad()

                    # Scales the loss, and calls backward() . to create scaled gradients
                    self.scaler.scale(loss).backward()
                    # Unscales gradients and calls or skips optimizer.step()
                    self.scaler.step(opt)
                    # Updates the scale for next iteration
                    self.scaler.update()  

            else:
                yp = model(X)
                loss = nn.CrossEntropyLoss()(yp,y)
                if opt:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]

        return total_err / len(loader.dataset), total_loss / len(loader.dataset)


    def epoch_adversarial(self, loader, model, attack, dataset, epsilon=0.1, num_iter=20, alpha=0.01, ratio=1, opt=None, **kwargs):
        """Adversarial training/evaluation epoch over the dataset"""
        total_loss, total_err = 0.,0.

        #ratio - some have adversarial example and other no
        no_adv = int(ratio * len(loader.dataset))
        no_clean = len(loader.dataset) - no_adv

        l_adv = [True] * no_adv 
        l_clean = [False] * no_clean 
        decision = l_adv + l_clean
        random.shuffle(decision)
        
        grad = 0 #for fgsm with gradient alignment

        # for fgsm free
        delta_real = None
        minibatch_replay = 4 
        counter_minibatch_replay = 0
        X_prev, y_prev =  None, None

        # for fgsm grad alignment
        #we use λ = 0.1 for the CIFAR-10 and λ = 0.5
        if dataset == "cifar10":
            grad_align_cos_lambda = 0.1 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "mnist":
            grad_align_cos_lambda = 0.5 # coefficient of the cosine gradient alignment regularizer
        elif dataset == "imageNet":
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer
        else: #svhn
            grad_align_cos_lambda = 0.2 # coefficient of the cosine gradient alignment regularizer

        ct = 0
        for X,y in loader:

            if attack == "fgsm_free" and X_prev is not None and counter_minibatch_replay % minibatch_replay != 0:  
                # take new inputs only each `minibatch_replay` iterations
                X, y = X_prev, y_prev # this way we ensure the same total number of images/batches/epcohs for free fgsm

            X,y = X.to(self.device), y.to(self.device)

            if decision[ct]:
                #adversarial example
                if attack == "fgsm": 
                    #adversarial examples fgsm
                    delta = self.fgsm(model, X, y, epsilon=epsilon, **kwargs) 

                elif attack == "fgsm_rs": 
                    #adversarial examples fgsm with random initialization of deltas
                    delta = self.fgsm_rs(model, X, y, epsilon=epsilon, **kwargs) 

                elif attack == "fgsm_free": 
                    #adversarial examples fgsm with random initialization of deltas
                    delta, delta_real = self.fgsm_free(model, X, y, delta_real, epsilon=epsilon, **kwargs) 
                    counter_minibatch_replay += 1

                    if counter_minibatch_replay % minibatch_replay == 0:
                        counter_minibatch_replay = 0
                        X_prev = X.clone()
                        y_prev = y.clone()

                elif attack == "pgd": 
                    #adversarial examples pgd_linf
                    delta = self.pgd_linf(model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha, **kwargs) 

                elif attack == "pgd_rs":
                    #adversarial examples pgd_linf with random initialization of deltas
                    delta = self.pgd_linf_rs(model, X, y, epsilon=epsilon, num_iter=num_iter, alpha=alpha, opt=opt, **kwargs) 

                elif attack == "fgsm_grad_align":
                    #adversarial examples fgsm with gradient alignment 
                    delta, grad = self.fgsm_grad_align(model, X, y, epsilon=epsilon, **kwargs) 

                else: 
                    print("wrong attack")
                    return -1

                X_input = X + delta 

            else:
                #clean data example
                X_input = X 


            if self.half_prec: 
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    yp = model(X_input)
                    loss = nn.CrossEntropyLoss().cuda()(yp,y)
            else:
                yp = model(X_input)
                loss = nn.CrossEntropyLoss()(yp,y)


            # gradient alignment 
            if decision[ct] and  attack == "fgsm_grad_align":
                # runs only if it's a adversarial exmaple and the attack is fgsm with gradient alignment 
                reg = torch.zeros(1).cuda(self.device)[0]  # for .item() to run correctly

                grad2 = self.get_input_grad(model, X, y, epsilon, delta_init='random_uniform', backprop=True)
                grads_nnz_idx = ((grad**2).sum([1, 2, 3])**0.5 != 0) * ((grad2**2).sum([1, 2, 3])**0.5 != 0)
                grad1, grad2 = grad[grads_nnz_idx], grad2[grads_nnz_idx]
                grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
                grad1_normalized = grad1 / grad1_norms[:, None, None, None]
                grad2_normalized = grad2 / grad2_norms[:, None, None, None]
                cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
                reg += grad_align_cos_lambda * (1.0 - cos.mean())

                loss += reg


            if opt: # to train
                if self.half_prec: 
                    opt.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()  
                else:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

      
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
            ct += 1

        return total_err / len(loader.dataset), total_loss / len(loader.dataset)


    def fgsm(self, model, X, y, epsilon=0.1):
        """ Construct FGSM adversarial examples on the examples X"""
        
        if self.half_prec: 
            delta = torch.zeros_like(X, requires_grad=True).cuda()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = nn.CrossEntropyLoss().cuda()(model(X + delta), y)  
            self.scaler.scale(loss).backward()  

        else:
            delta = torch.zeros_like(X, requires_grad=True)
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()

        return epsilon * delta.grad.detach().sign()


    def fgsm_rs(self, model, X, y, epsilon=0.1, alpha=0.375):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
        delta.requires_grad = True
        output = model(X + delta)
        
        loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
        delta = delta.detach()
        
        return delta


    def fgsm_free(self, model, X, y, delta, epsilon=0.1, alpha=0.375):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        if delta is None:
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
    
        delta.requires_grad = True
        output = model(X + delta[:X.size(0)])
        
        loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:X.size(0)] = torch.max(torch.min(1-X, delta.data[:X.size(0)]), 0-X)
        delta_return = delta.detach()
        
        return delta_return[:X.size(0)], delta


    def fgsm_grad_align(self, model, X, y, epsilon=0.1, alpha=0.375):
        """ Construct FGSM adversarial examples on the examples X with unform random initialization"""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)# .cuda()
        delta.requires_grad = True
        output = model(X + delta)
        
        loss = F.cross_entropy(output, y) # WHERE SHOUDL WE USE THE CROSS ENTROPY or  CrossEntropyLoss???
        grad = torch.autograd.grad(loss, delta, create_graph=True)[0]

        loss.backward()

        #grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
        
        delta = delta.detach()
        grad = grad.detach()

        
        return delta, grad


    def pgd_linf(self, model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
        """ Construct FGSM adversarial examples on the examples X"""
        #print(epsilon)
        if self.half_prec: 
            if randomize:
                delta = torch.rand_like(X, requires_grad=True).cuda()
                delta.data = delta.data * 2 * epsilon - epsilon
            else:
                delta = torch.zeros_like(X, requires_grad=True).cuda()

            for t in range(num_iter):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = nn.CrossEntropyLoss().cuda()(model(X + delta), y)
                
                self.scaler.scale(loss).backward()  
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
                delta.grad.zero_()                   
        else:

            if randomize:
                delta = torch.rand_like(X, requires_grad=True)
                delta.data = delta.data * 2 * epsilon - epsilon
            else:
                delta = torch.zeros_like(X, requires_grad=True)
                
            for t in range(num_iter):
                loss = nn.CrossEntropyLoss()(model(X + delta), y)
                loss.backward()
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
                delta.grad.zero_()

        return delta.detach()

    def pgd_linf_rs(self, model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False, opt=None):
        """ Construct FGSM adversarial examples on the examples X"""

        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)

        for _ in range(num_iter):
            delta.requires_grad = True
            output = model(X + delta)

            loss =  nn.CrossEntropyLoss()(output, y)
            if opt:
                opt.zero_grad()
            loss.backward()

            grad = delta.grad.detach()
            I = output.max(1)[1] == y
            delta.data[I] = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)[I]
            delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]

        return delta.detach()


    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


    def pgd_l2(self, model, X, y, epsilon, alpha, num_iter):
        delta = torch.zeros_like(X, requires_grad=True)

        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data += alpha*delta.grad.detach() / self.norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
            delta.data *= epsilon / self.norms(delta.detach()).clamp(min=epsilon)
            delta.grad.zero_()
            
        return delta.detach()


    def get_input_grad(self, model, X, y, eps, delta_init='none', backprop=False):
        if delta_init == 'none':
            delta = torch.zeros_like(X, requires_grad=True)
        elif delta_init == 'random_uniform':
            delta = torch.zeros_like(X).uniform_(-eps, eps)
            delta.requires_grad = True
            #delta = self.get_uniform_delta(X.shape, eps, requires_grad=True)
        elif delta_init == 'random_corner':
            delta = torch.zeros_like(X).uniform_(-eps, eps)
            delta.requires_grad = True
            #delta = self.get_uniform_delta(X.shape, eps, requires_grad=True)
            delta = eps * torch.sign(delta)
        else:
            raise ValueError('wrong delta init')

        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]

        if not backprop:
            grad, delta = grad.detach(), delta.detach()
        return grad


    def get_uniform_delta(self, shape, eps, requires_grad=True):
        delta = torch.zeros(shape) #.cuda()
        delta.uniform_(-eps, eps)
        delta.requires_grad = requires_grad
        return delta


    def l2_norm_batch(self, v):
        norms = (v ** 2).sum([1, 2, 3]) ** 0.5
        return norms


    def testModel_logs(self, dataset_name, models_name, iteration, alg, ratio ,epsilon, numIt, alpha, ratioADV, trainTime):

        self.model.eval() # evaluate the model
        
        list_to_write = []

        #test the accuracy of the model
        t3 = time.time()
        test_err, test_loss = self.epoch(self.loader.test_loader, self.model)
        testTime = time.time() - t3

        _ratio = 0.0 if isinstance(ratio, list) else ratio # only used for standard training 
            

        str_write = str(iteration) + "," + alg + "," + str(_ratio) + "," + str(epsilon) + "," + str(numIt) + "," + str(alpha) + "," + str(ratioADV) +  \
                            ",std,0,0,0," + str(test_err) + "," + str(test_loss) + "," + str(testTime) + "," + str(trainTime) + "\n"
        list_to_write.append(str_write)


        #test the adversarial accuracy of the model
        if dataset_name == "cifar10":
            eps_test_list =  [2, 4, 8, 12, 16]
        elif dataset_name ==   "mnist": #mnist
            eps_test_list =  [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        elif dataset_name == "imageNet":
            eps_test_list =  [2, 4, 8]
        else: #if dataset_name ==   "svhn":
            eps_test_list =  [2, 4, 8, 12]

        num_iterTest_list = [20]
        alpha_test_list = [0.01]

        # testing the final model using PGD
        for i, eps_test in enumerate(eps_test_list):
            for num_iterTest in num_iterTest_list:
                for alpha_test in alpha_test_list:
                    if dataset_name == "mnist": #mnist
                        _eps_test = eps_test
                    #if dataset_name == "cifar10":
                    #    _eps_test = eps_test/255.0
                    else: #svhn or cifar10 or imageNet
                        _eps_test = eps_test/255.0

                    t4 = time.time()
                    adv_err, adv_loss = self.epoch_adversarial(self.loader.test_loader, self.model, "pgd", "", _eps_test, num_iterTest, alpha_test, 1)
                    advTestTime = time.time() - t4

                    str_write = str(iteration) + "," + alg + "," + str(_ratio) + "," + str(epsilon) + "," + str(numIt) + "," + str(alpha) + "," + str(ratioADV) + \
                                    ",pgd," + str(_eps_test) + "," + str(num_iterTest) + "," + str(alpha_test) + "," + \
                                    str(adv_err) + "," + str(adv_loss) + "," + str(advTestTime) + "," + str(trainTime) + "\n"

                    list_to_write.append(str_write)


        #write logs
        if isinstance(ratio, list):
            for _ratio_2_test in ratio:
                if _ratio_2_test < 1.0:
                    filename = "./logs1/logs_" + models_name + "_ratio" + str(_ratio_2_test) + ".txt"
                else:
                    filename = "./logs/logs_" + models_name + ".txt"

                if iteration == 0:
                    f = open(filename, "w")
                    f.write("it,alg,ratio,epsilon,numIt,alpha,ratioAdv,algTest,epsilonTest,numItTest,alphaTest,Error,Loss,testingTime,trainTime\n")
                else:
                    f = open(filename, "a")
        

                for str_write in list_to_write:
                    f.write(str_write)

        else:

            filename = "./logs/logs_" + models_name + ".txt"
            if iteration == 0:
                f = open(filename, "w")
                f.write("it,alg,ratio,epsilon,numIt,alpha,ratioAdv,algTest,epsilonTest,numItTest,alphaTest,Error,Loss,testingTime,trainTime\n")
            else:
                f = open(filename, "a")


            for str_write in list_to_write:
                f.write(str_write)

        f.close()
        self.model.train() # go back to train mode


        return


class model(trainModel):
    def __init__(self, dataset, dataset_name, device, devices_id, lr=0.1, momentum=0, lr_adv=0.1, momentum_adv=0, batch_adv=100, half_prec=False):
        self.loader = dataset
        self.dataset_name = dataset_name 
        self.lr = lr 
        self.momentum = momentum 

        self.lr_adv = lr_adv 
        self.momentum_adv = momentum_adv 
        self.batch_adv = batch_adv

        self.devices_id = devices_id


        if self.dataset_name != "imageNet":
            half_prec = False

        super().__init__(device, half_prec=half_prec) #initialize the datasets
    
        self.model = self.resetModel() #initialize model

        dampening=0
        weight_decay=0 if self.dataset_name != "imageNet" else 0.0001
        nesterov=False
        self.opt = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        # hyper-parameters clean data: learning rate, momentum, dampening, weight_decay, nesterov
        # hyper-parameters FGSM: epsilon, ratio
        # hyper-parameters PGD: epsilon, num_iter, alpha, ratio

        if self.dataset_name == "svhn" or self.dataset_name == "imageNet":
            # we only parallelize these 2 datasets
            self.model = self.parallelizeModel(self.model)


    def run(self, modelName, loadModel=0, iterations=10, stop="epochs"):    
        
        self.model.train()
        if stop == "epochs":
            return self._train_epochs(modelName, loadModel, iterations=iterations)
        else:
            return self._train_time(modelName, loadModel, stop_time=iterations)


    def _train_epochs(self, modelName, loadModel=0, iterations=10):
        ''' uploads the models or trains it from scratch'''
        path = "./models/" + modelName
        if "cifar10" in modelName:
            _dataset = "cifar10"
        elif "mnist" in modelName:
            _dataset = "mnist"
        elif "imageNet" in modelName:
            _dataset = "imageNet"
        else: #svhn
            _dataset = "svhn"

        # loadModel
        # 0 -> no checkpointing
        # 1 -> checkpointing and running until the end (only used fo standard training)
        # 2 -> checkpointing and early stopping (only used fo standard training to differentiate the full standard training and pre-training) 

        aux = modelName.split("_")

        if  "robust" in aux[1] and aux[2] == "FGSM":
            _ratio =  float(aux[4][5:])
            ratio_adv = float(aux[5][8:])

            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            else: #svhn or cifar10 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            print('train with clean data and then with fgsm (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])))


            if loadModel!=0 and _ratio < 1.0:
                #self.standard_train_save(self.model, modelName, self.loader, _dataset, self.opt, iterations=iterations)
                trainTime, train_err, train_loss = self.standard_fgsm_train_load(self.model, modelName, self.loader, _dataset, self.opt, ratio=_ratio, 
                        eps_train=_eps_train, iterations=iterations, ratio_adv=ratio_adv, lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)
                
            else:
                trainTime, train_err, train_loss = self.standard_fgsm_train(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, ratio=_ratio, 
                        eps_train=_eps_train, iterations=iterations, ratio_adv=ratio_adv, lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)

        elif "robust" in aux[1] and aux[2] == "PGD":
            _ratio = float(aux[6][5:])
            ratio_adv = float(aux[7][8:])

            _num_iterTrain = int(aux[4][5:])
            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            else: #svhn or cifar10 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            _alpha_train =float( aux[5][5:])

            print('train with clean data and then with pgd (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])) + ', no_ite=' + str(_num_iterTrain) + ', alpha=' + str(_alpha_train))
            
            if loadModel!=0 and _ratio < 1.0:
                #self.standard_train_save(self.model, modelName, self.loader, _dataset, self.opt, iterations=iterations)
                trainTime, train_err, train_loss = self.standard_pgd_train_load(self.model, modelName, self.loader, _dataset, self.opt, ratio=_ratio, 
                        num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, iterations=iterations, ratio_adv=ratio_adv,
                        lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)
            else:

                trainTime, train_err, train_loss = self.standard_pgd_train(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, ratio=_ratio, 
                        num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, iterations=iterations, ratio_adv=ratio_adv,
                        lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)


        elif "robust" in aux[1] and aux[2] ==  "FGSMrs":
        # "model_robust_FGSMrs_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt"
            print("WARINING: FGSMrs doesn't allow clean+adversarial training yet!")
            _ratio =  float(aux[4][5:])
            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            else: #svhn or cifar10 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            print("train with fgsm random initialization with ratio " + str(_ratio) + " and epsilon " + str(float(aux[3][3:])))

            trainTime, train_err, train_loss = self.adversarial_train_fgsm_rs(self.model, modelName, self.loader, _dataset, saveModel=True, ratio=_ratio, 
                        eps_train=_eps_train, iterations=iterations, lr=self.lr_adv, momentum=self.momentum_adv)


        elif "robust" in aux[1] and aux[2] ==  "FGSMfree":
        # "model_robust_FGSMfree_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt"
            print("WARINING: FGSMfree doesn't allow clean+adversarial training yet!")

            _ratio =  float(aux[4][5:])
            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            else: #svhn or cifar10 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            print("train with free fgsm with ratio " + str(_ratio) + " and epsilon " + str(float(aux[3][3:])))

            trainTime, train_err, train_loss = self.adversarial_train_fgsm_free(self.model, modelName, self.loader, _dataset, saveModel=True, ratio=_ratio, 
                        eps_train=_eps_train, iterations=iterations, lr=self.lr_adv, momentum=self.momentum_adv)


        elif "robust" in aux[1] and aux[2] ==  "FGSMgradAlign":
            print("WARINING: FGSMgradAlign doesn't allow clean+adversarial training yet!")

            _ratio =  float(aux[4][5:])
            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            #    _eps_train = float(aux[3][3:])/255.0
            else: #svhn or cifar10 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            print("train with fgsm gradient alignment with ratio " + str(_ratio) + " and epsilon " + str(float(aux[3][3:])))

            trainTime, train_err, train_loss = self.adversarial_train_fgsm_grad_align(self.model, modelName, self.loader, _dataset, saveModel=True, 
                        ratio=_ratio, eps_train=_eps_train, iterations=iterations, lr=self.lr_adv, momentum=self.momentum_adv)


        elif "robust" in aux[1] and aux[2] == "PGDrs":
            print("WARINING: PGDrs doesn't allow clean+adversarial training yet!")

        # "model_robust_PGDrs_eps" + str(eps_train) + "_numIt" + str(num_iterTrain) + "_alpha" + str(alpha_train) + "_ratio" + str(ratio) + ".pt"
            _ratio =  float(aux[6][5:])
            _num_iterTrain = int(aux[4][5:])
            if "mnist" in modelName:
                _eps_train = float(aux[3][3:])
            #if "cifar10" in modelName:
            #    _eps_train = float(aux[3][3:])/255.0
            else: #svhn or cifar10 or imageNet
                _eps_train = float(aux[3][3:])/255.0

            _alpha_train = float(aux[5][5:])

            print("train with pgd random initialization with ratio " + str(_ratio) + " and epsilon " + str(float(aux[3][3:])) + " no of iterations " + str(_num_iterTrain) + " and alpha " + str(_alpha_train))

            trainTime, train_err, train_loss = self.adversarial_train_pgd_rs(self.model, modelName, self.loader, _dataset, saveModel=True, ratio=_ratio, 
                        num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, iterations=iterations, lr=self.lr_adv, momentum=self.momentum_adv)


        else:
            # loadModel
            # 0 -> no checkpointing
            # 1 -> checkpointing and running until the end (only used fo standard training)
            # 2 -> checkpointing and early stopping (only used fo standard training to differentiate the full standard training and pre-training) 

            # model_std_train.pt
            if loadModel==1:
                #train all model with checkpointing and running until the end (only used fo standard training)
                trainTime, train_err, train_loss = self.standard_train_save(self.model, modelName, self.loader, _dataset, self.opt, iterations=iterations, stop=False)
            elif loadModel==2:
                #train all model with checkpointing and early stopping (only used fo standard training to differentiate the full standard training and pre-training) 
                trainTime, train_err, train_loss = self.standard_train_save(self.model, modelName, self.loader, _dataset, self.opt, iterations=iterations, stop=True)
            else:
                trainTime, train_err, train_loss = self.standard_train(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, iterations=iterations)

        return trainTime, train_err, train_loss
        

    def _train_time(self, modelName, loadModel=False, stop_time=100):
        ''' uploads the models or trains it from scratch'''
        path = "./models/" + modelName

        if "cifar10" in modelName:
            _dataset = "cifar10"
        elif "mnist" in modelName:
            _dataset = "mnist"
        elif "imageNet" in modelName:
            _dataset = "imageNet"
        else: #svhn
            _dataset = "svhn"

        if loadModel and os.path.exists(path):
            #load model
            self.model.load_state_dict(torch.load(path))
            return None, None, None

        else:
            #otherwise train it
            aux = modelName.split("_")
            if  "robust" in aux[1] and aux[2] == "FGSM":
                _ratio =  float(aux[4][5:])
                ratio_adv = float(aux[5][8:])

                if "mnist" in modelName:
                    _eps_train = float(aux[3][3:])
                else: #svhn or cifar10 or imageNet
                    _eps_train = float(aux[3][3:])/255.0

                print('train (stop codition time) with clean data and then with fgsm (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])))

                #trainTime, train_err, train_loss = self.adversarial_train_fgsm_time(self.model, modelName, self.loader, _dataset, saveModel=True, 
                #            ratio=_ratio, eps_train=_eps_train, stop_time=stop_time, lr=self.lr_adv, momentum=self.momentum_adv)

                #trainTime, train_err, train_loss = self.standard_fgsm_train_time(self.model, modelName, self.loader, _dataset, saveModel=True, ratio=_ratio, 
                #            eps_train=_eps_train, stop_time=stop_time, ratio_adv=ratio_adv, lr=self.lr, momentum=self.momentum, 
                #            lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)
                trainTime, train_err, train_loss = self.standard_fgsm_train_time(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, ratio=_ratio, 
                            eps_train=_eps_train, stop_time=stop_time,ratio_adv=ratio_adv,lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)

            elif "robust" in aux[1] and aux[2] == "PGD":
            # "model_robust_PGD_eps" + str(eps_train) + "_numIt" + str(num_iterTrain) + "_alpha" + str(alpha_train) + "_ratio" + str(ratio) + ".pt"
                ratio_adv = float(aux[7][8:])
                _ratio = float(aux[6][5:])

                _num_iterTrain = int(aux[4][5:])

                if "mnist" in modelName:
                    _eps_train = float(aux[3][3:])
                else: #svhn or cifar10 or imageNet
                    _eps_train = float(aux[3][3:])/255.0

                _alpha_train =float( aux[5][5:])

                print('train (stop codition time) with clean data and then with pgd (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])) + ', no_ite=' + str(_num_iterTrain) + ', alpha=' + str(_alpha_train))

                #trainTime, train_err, train_loss = self.adversarial_train_pgd_time(self.model, modelName, self.loader, _dataset, saveModel=True, ratio=_ratio, 
                #            num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, stop_time=stop_time, lr=self.lr_adv, momentum=self.momentum_adv)
                #trainTime, train_err, train_loss = self.standard_pgd_train_time(self.model, modelName, self.loader, _dataset, saveModel=True, ratio=_ratio, 
                #            num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, stop_time=stop_time, ratio_adv=ratio_adv, lr=self.lr, momentum=self.momentum, lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)
                trainTime, train_err, train_loss = self.standard_pgd_train_time(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, ratio=_ratio, 
                            num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, stop_time=stop_time, ratio_adv=ratio_adv, lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)
            

            elif "robust" in aux[1] and aux[2] ==  "FGSMrs":
                #print("WARINING: FGSMrs doesn't allow clean+adversarial training yet!")

                _ratio =  float(aux[4][5:])
                ratio_adv = float(aux[5][8:])

                if "mnist" in modelName:
                    _eps_train = float(aux[3][3:])
                else: #svhn or cifar10 or imageNet
                    _eps_train = float(aux[3][3:])/255.0

                print('train (stop codition time) with clean data and then with fgsmRs (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])) + ")")

                trainTime, train_err, train_loss = self.standard_fgsmRs_train_time(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, ratio=_ratio, 
                            eps_train=_eps_train, stop_time=stop_time,ratio_adv=ratio_adv,lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)

                #trainTime, train_err, train_loss = self.adversarial_train_fgsm_rs_time(self.model, modelName, self.loader, _dataset, saveModel=True, 
                #            ratio=_ratio, eps_train=_eps_train, stop_time=stop_time, lr=self.lr_adv, momentum=self.momentum_adv)


            elif "robust" in aux[1] and aux[2] ==  "FGSMfree":

                _ratio =  float(aux[4][5:])
                ratio_adv = float(aux[5][8:])

                if "mnist" in modelName:
                    _eps_train = float(aux[3][3:])
                else: #svhn or cifar10 or imageNet
                    _eps_train = float(aux[3][3:])/255.0

                print('train (stop codition time) with clean data and then with fgsmFree (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])) + ")")
                
                trainTime, train_err, train_loss = self.standard_fgsmFree_train_time(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, ratio=_ratio, 
                            eps_train=_eps_train, stop_time=stop_time,ratio_adv=ratio_adv,lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)

                #trainTime, train_err, train_loss = self.adversarial_train_fgsm_free_time(self.model, modelName, self.loader, _dataset, saveModel=True, 
                #            ratio=_ratio, eps_train=_eps_train, stop_time=stop_time, lr=self.lr_adv, momentum=self.momentum_adv)


            elif "robust" in aux[1] and aux[2] ==  "FGSMgradAlign":

                _ratio =  float(aux[4][5:])
                ratio_adv = float(aux[5][8:])


                if "mnist" in modelName:
                    _eps_train = float(aux[3][3:])
                else: #svhn or cifar10 or imageNet
                    _eps_train = float(aux[3][3:])/255.0
                print('train (stop codition time) with clean data and then with fgsm gradient alignment (ratioTime=' + str(_ratio*100) + '%, ratioAdv=' + str(ratio_adv*100) + '%, eps=' + str(float(aux[3][3:])) + ")")


                trainTime, train_err, train_loss = self.standard_fgsmGrad_align_train_time(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, ratio=_ratio, 
                            eps_train=_eps_train, stop_time=stop_time,ratio_adv=ratio_adv,lr_adv=self.lr_adv, momentum_adv=self.momentum_adv)


            elif "robust" in aux[1] and aux[2] == "PGDrs":
                print("WARINING: PGDrs doesn't allow clean+adversarial training yet!")

                _ratio =  float(aux[6][5:])
                _num_iterTrain = int(aux[4][5:])

                if "mnist" in modelName:
                    _eps_train = float(aux[3][3:])
                #if "cifar10" in modelName:
                #    _eps_train = float(aux[3][3:])/255.0
                else: #svhn or cifar10 or imageNet
                    _eps_train = float(aux[3][3:])/255.0

                _alpha_train = float(aux[5][5:])

                print("train (stop codition time) with pgd random initialization with ratio " + str(_ratio) + " and epsilon " + str(float(aux[3][3:])) + " no of iterations " + str(_num_iterTrain) + " and alpha " + str(_alpha_train))

                trainTime, train_err, train_loss = self.adversarial_train_pgd_rs_time(self.model, modelName, self.loader, _dataset, saveModel=False, 
                            ratio=_ratio, num_iterTrain=_num_iterTrain, eps_train=_eps_train, alpha_train=_alpha_train, stop_time=stop_time, 
                            lr=self.lr_adv, momentum=self.momentum_adv)


            else:
            # model_std_train.pt

                print("standard training (stop codition time) ")
                #trainTime, train_err, train_loss = self.standard_train_time(self.model, modelName, self.loader, _dataset, saveModel=True, stop_time=stop_time, 
                #            lr=self.lr, momentum=self.momentum)
                trainTime, train_err, train_loss = self.standard_train_time(self.model, modelName, self.loader, _dataset, self.opt, saveModel=False, stop_time=stop_time)

        return trainTime, train_err, train_loss
        

    def testModel(self):
        '''test the model at the end to evaluate standard accuracy'''
        # switch to evaluate mode
        self.model.eval()

        t3 = time.time()
        test_err, test_loss = self.epoch(self.loader.test_loader, self.model)
        testTime = time.time() - t3

        return (testTime, test_err, test_loss)


    def testModel_adversarial_pgd_all(self):
        '''test the model with adversarial examples at the end to evaluate adversarial accuracy
            using different sets of hyperparameter'''
        # switch to evaluate mode
        self.model.eval()

        eps_test_list =  [0.01, 0.05, 0.1, 0.2]
        num_iterTest_list = [10, 20]
        alpha_test_list = [0.01]

        advTestTime_pgd = []
        adv_loss_pgd = []
        adv_err_pgd = []

        # testing the final model using PGS
        for eps_test in eps_test_list:
            for num_iterTest in num_iterTest_list:
                for alpha_test in alpha_test_list:
                    print("PGD test epsilon test= " + str(eps_test) + "  num_iterTest = " + str(num_iterTest)  + "  alpha_test = " + str(alpha_test))

                    t4 = time.time()
                    adv_err, adv_loss = self.epoch_adversarial(self.loader.test_loader, self.model, "pgd", "", eps_test, num_iterTest, alpha_test, 1)
                    advTestTime = time.time() - t4

                    advTestTime_pgd.append(advTestTime)
                    adv_loss_pgd.append(adv_loss)
                    adv_err_pgd.append(adv_err)

        return (advTestTime_pgd, adv_err_pgd, adv_loss_pgd)


    def testModel_adversarial_pgd(self, eps_test, num_iterTest, alpha_test):
        '''test the model with adversarial examples at the end to evaluate adversarial accuracy
            using fixing hyperparameter''' 
        # switch to evaluate mode
        self.model.eval()

        #print("PGD test epsilon test= " + str(eps_test) + "  num_iterTest = " + str(num_iterTest)  + "  alpha_test = " + str(alpha_test)
        t4 = time.time()
        adv_err, adv_loss = self.epoch_adversarial(self.loader.test_loader, self.model, "pgd", "", eps_test, num_iterTest, alpha_test, 1)
        advTestTime = time.time() - t4

        return (advTestTime, adv_err, adv_loss)


    def saveResults(self, pathFile, data):
        self.writeResult(pathFile, data)


    def parallelizeModel(self, model):
        if self.devices_id is not None: # parallelize the job 
            model = nn.DataParallel(model, device_ids = self.devices_id)
        return model.to(self.device)
            

    def resetModel(self, ):
        if self.dataset_name == "mnist":
            return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                                        Flatten(),
                                        nn.Linear(7*7*64, 100), nn.ReLU(),
                                        nn.Linear(100, 10)).to(self.device)

        elif self.dataset_name == "cifar10":
            return nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
                        nn.BatchNorm2d(64),

                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
                        nn.BatchNorm2d(128),

                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
                        nn.BatchNorm2d(256),

                        nn.Flatten(), 
                        nn.Linear(256*4*4, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 10)).to(self.device)

        elif self.dataset_name == "svhn":
            model = resnet18()
            #model = resnet50()
            return model.to(self.device)

        elif self.dataset_name == "imageNet":
            option = 4

            ## PYTORCH 
            if option==0: model = resnet18()
            elif option==1: model = resnet50()

            ###
            # code taken from https://github.com/xternalz/WideResNet-pytorch
            ###
            elif option==2:
                depth = 28
                num_classes = 1000
                widen_factor = 2
                droprate = 0.0
                model = WideResNet(depth, num_classes, widen_factor, droprate)


            ###
            # code taken from https://github.com/tml-epfl/understanding-fast-adv-training
            ###
            elif option==3:
                num_classes = 1000
                model = PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=num_classes, cuda=True, half_prec=self.half_prec)


            ###
            # code taken from https://github.com/landskape-ai/ImageNet-Downsampled
            ###
            elif option==4:
                depth = 50
                num_classes = 1000
                if depth == 18:
                    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, depth)
                elif depth == 34:
                    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, depth)
                elif depth == 50:
                    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, depth)
                elif depth == 101:
                    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, depth)
                else:
                    model = ResNet(BasicBlock, [2, 2, 2, 2], 1000, 18)
                

            return model.to(self.device)
            
        else:
            print("error on model")


def main(modelName, dataset_name, iterations=10, stop="epochs", device=None, devices_id=None, lr=0.1, momentum=0, batch=100, lr_adv=0.1, momentum_adv=0, batch_adv=100, ratio=1, half_prec=False, preTrain=False):
    
    dataset_loader = dataset(dataset_name=dataset_name, batch_size = batch, batch_size_adv = batch_adv, ratio=ratio)
    model_cnn = model(dataset_loader, dataset_name, device, devices_id, lr, momentum, lr_adv, momentum_adv, batch_adv, half_prec=half_prec)
    trainTime, train_err, train_loss = model_cnn.run(modelName, loadModel=preTrain, iterations=iterations, stop=stop) # train

    if trainTime == -1: return # pre trained model

    print("start testing ")
    testTime, test_err, test_loss = model_cnn.testModel()
    
    data = [None]*3
    data[0] = trainTime, train_err, train_loss 
    data[1] = testTime, test_err, test_loss

    if dataset_name == "cifar10":
        eps_test_list =  [2, 4, 8, 12, 16]
    elif dataset_name ==   "mnist": #mnist
        eps_test_list =  [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    elif dataset_name == "imageNet":
        eps_test_list =  [2, 4, 8]
    else: #if dataset_name ==   "svhn":
        #eps_test_list =  [2, 4, 8, 12, 16]
        eps_test_list =  [4, 8, 12, 16]

    num_iterTest_list = [10, 20]
    num_iterTest_list = [20]
    alpha_test_list = [0.01]

    #advTestTime_pgd = []
    #adv_loss_pgd = []
    #adv_err_pgd = []

    # testing the final model using PGD
    for i, eps_test in enumerate(eps_test_list):
        for num_iterTest in num_iterTest_list:
            for alpha_test in alpha_test_list:
                if dataset_name == "mnist": #mnist
                    _eps_test = eps_test
                else: #svhn or cifar10 or imageNet
                    _eps_test = eps_test/255.0

                advTestTime, adv_err, adv_loss = model_cnn.testModel_adversarial_pgd(_eps_test, num_iterTest, alpha_test)
                data[2] = advTestTime, adv_err, adv_loss

                name = "./times/" + modelName + "_epsTest" + str(eps_test) + "_numItTest" + str(num_iterTest) + "_alphatest" + str(alpha_test) + ".csv"
                
                model_cnn.saveResults( name, data)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('True', 'yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    # "model_robust_FGSMrs_eps" + str(eps_train) + "_ratio" + str(ratio) + ".pt"
    # "model_robust_PGD_eps" + str(eps_train) + "_numIt" + str(num_iterTrain) + "_alpha" + str(alpha_train) + "_ratio" + str(ratio) + ".pt"

    # model_std_train.pt

    parser = argparse.ArgumentParser()


    parser.add_argument('--type', type=str, help='type of traing', default="std", choices=['std', 'robust'])
    parser.add_argument('--alg', type=str, help='path to store the model', default="pgd", choices=['pgd', 'fgsm', 'pgd_rs', 'fgsm_rs', 'fgsm_free', 'fgsm_grad_align'])

    parser.add_argument('--ratio_adv', type=float, help='percentage of data to train adversarial just for std+adv', default=1.0)
    parser.add_argument('--ratio', type=float, help='percentage of data to train adversarial', default=1.0)
    parser.add_argument('--epsilon', type=float, help='epsilon bound', default=0.1)

    parser.add_argument('--num_iter', type=int, help='number of iterations for pgd ', default=10)
    parser.add_argument('--alpha', type=float, help='alpha', default=0.01)
    
    parser.add_argument('--dataset', type=str, help='dataset', default="mnist", choices=['mnist', 'cifar10', 'imageNet', 'svhn'])

    parser.add_argument('--stop', type=str, help='stop condition', default="epochs", choices=['epochs', 'time'])
    parser.add_argument('--stop_val', type=int, help='number of epochs or training time', default=10)

    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--momentum', type=float, help='momentum', default=0.0)
    parser.add_argument('--batch', type=int, help='batch', default=100)

    parser.add_argument('--lr_adv', type=float, help='learning rate adv training', default=0.01)
    parser.add_argument('--momentum_adv', type=float, help='momentum adv training', default=0.0)
    parser.add_argument('--batch_adv', type=int, help='batch adv training', default=100)

    parser.add_argument('--workers', type=str, help='GPU workers', default="0")
    parser.add_argument('--half_prec', type=str2bool, help='half precision', default=False)
    parser.add_argument('--preTrain', type=int, help='preTrain model', default=0, choices=[0,1,2])

    args = parser.parse_args()

    dataset_name = args.dataset
    if args.type == "std":
        args.ratio = 0.0
        args.ratio_adv = 0.0

    modelName = "model" + dataset_name

    if args.type == "std":
        modelName += "_std_train"

    else: #robust
        if "fgsm" in args.alg:
            modelName += "_robust_FGSM"
            if "rs" in args.alg:
                modelName += "rs"
            elif 'free' in args.alg:
                modelName += "free"
            elif 'grad_align' in args.alg:
                modelName += "gradAlign"

            modelName += "_eps" + str(args.epsilon) + "_ratio" + str(args.ratio) + "_ratioadv" + str(args.ratio_adv)

        else: #pgd
            modelName += "_robust_PGD"
            if "rs" in args.alg:
                modelName += "rs"
            elif 'free' in args.alg:
                modelName += "free"
            elif 'grad_align' in args.alg:
                modelName += "gradAlign"

            modelName += "_eps" + str(args.epsilon) + "_numIt" + str(args.num_iter) + "_alpha" + str(args.alpha) + "_ratio" + str(args.ratio) + "_ratioadv" + str(args.ratio_adv)

    modelName += "_lr" + str(args.lr) + "_momentum" + str(args.momentum) + "_batch" + str(args.batch)
    modelName += "_lrAdv" + str(args.lr_adv) + "_momentumAdv" + str(args.momentum_adv) + "_batchAdv" + str(args.batch_adv)

    #device = torch.device("cuda:" + str(args.workers) if torch.cuda.is_available() else "cpu")
    
    devices_id = [0]
    if args.dataset == 'imageNet' or args.dataset == 'svhn':
        aux_workers = args.workers.split(",")
        if len(aux_workers) > 1:
            devices_id = []
            for aux_w in  aux_workers:
                devices_id.append(int(aux_w))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    toRun= False
    if args.type == "std":
        if args.preTrain==0: # no checnkpointing 
            if not os.path.isfile("./logs/logs_" + modelName + ".txt"): toRun= True

        elif args.preTrain==1: # checkpointing and running until the end (only used fo standard training)
            if not os.path.isfile("./logs/logs_" + modelName + ".txt"): toRun= True

            if not os.path.isfile("./models/" + modelName + "_ratio0.3.pt"): toRun= True
            if not os.path.isfile("./logs1/logs_" + modelName + "_ratio0.3.txt"): toRun= True

            if not os.path.isfile("./models/" + modelName + "_ratio0.5.pt"): toRun= True
            if not os.path.isfile("./logs1/logs_" + modelName + "_ratio0.5.txt"): toRun= True

            if not os.path.isfile("./models/" + modelName + "_ratio0.7.pt"): toRun= True
            if not os.path.isfile("./logs1/logs_" + modelName + "_ratio0.7.txt"): toRun= True
        
        else: #checkpointing and early stopping (only used fo standard training to differentiate the full standard training and pre-training) 
            if not os.path.isfile("./models/" + modelName + "_ratio0.3.pt"): toRun= True
            if not os.path.isfile("./logs1/logs_" + modelName + "_ratio0.3.txt"): toRun= True

            if not os.path.isfile("./models/" + modelName + "_ratio0.5.pt"): toRun= True
            if not os.path.isfile("./logs1/logs_" + modelName + "_ratio0.5.txt"): toRun= True

            if not os.path.isfile("./models/" + modelName + "_ratio0.7.pt"): toRun= True
            if not os.path.isfile("./logs1/logs_" + modelName + "_ratio0.7.txt"): toRun= True

    else: # fgsm and pgd
        if not os.path.isfile("./logs/logs_" + modelName + ".txt"): toRun= True
    

    if toRun: # config doesnt exist in the VM
        if os.path.isfile("configs_run.txt"):
            with open("configs_run.txt", "r") as fl:
                for line in fl:
                    if "logs_" + modelName + ".txt" == line[:-1]: 
                        toRun= False
                        print("in file")
                        break


    if toRun:
        main(modelName, dataset_name, args.stop_val, args.stop, device, devices_id, args.lr, args.momentum, args.batch,  args.lr_adv, args.momentum_adv, 
                            args.batch_adv, args.ratio, args.half_prec, args.preTrain)  

        if os.path.isfile("configs_run.txt"):
            with open("configs_run.txt", "a") as fl:
                fl.write("logs_" + modelName + ".txt\n")

    else:
        print("config exists (no checkpointing is needed)")

    #main(modelName, dataset_name, args.stop_val, args.stop, device, args.lr, args.momentum, args.batch,  args.lr_adv, args.momentum_adv, args.batch_adv, args.ratio)
    

 

#ython3 train.py --type=std --dataset=imageNet --stop=time --stop_val=21600 --lr=0.1 --momentum=0.9 --batch=512 --lr_adv=0 --momentum_adv=0 --batch_adv=0 --half_prec=True --workers=0,1,2,3
#python3 train.py --type=std --dataset=imageNet --stop=time --stop_val=43200 --lr=0.1 --momentum=0.9 --batch=512 --lr_adv=0 --momentum_adv=0 --batch_adv=0 --half_prec=True --workers=0

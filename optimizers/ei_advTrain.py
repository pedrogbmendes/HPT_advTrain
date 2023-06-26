#!/usr/bin/env python
import argparse
import sys
import numpy as np
import collections
import csv
import math, time, torch
import os
#from robo.fmin import fabolas
from botorch.models.transforms.outcome import Standardize

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import EqualsCondition, LessThanCondition, GreaterThanCondition, OrConjunction, NotEqualsCondition, AndConjunction

from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement

from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


network = 'imageNet'
network = 'cifar10'
network = 'svhn'
#network = 'mnist'


csv.field_size_limit(sys.maxsize)

listConfig = []

acqFunc = "random"
acqFunc = "EI"


torch.set_printoptions(precision=7, sci_mode=True)
tkwargs = {
    "dtype": torch.double,
    #"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}

if network == 'imageNet':
    Bound_epsilon = float(2)
elif network == 'cifar10':
    Bound_epsilon = float(12)
elif network == 'svhn':
    Bound_epsilon = float(8)
elif network == 'mnist':
    Bound_epsilon = float(0.3)
else:
    print("wrong dataset")
    sys.exit()



class model_benchmark():

    _optimal_value = 0.0

    def __init__(self, configspace):

        self.a = 0.5
        self.factor = 10000.0
        self.configspace= configspace

        #datasetName = "./hpbandster/workers/data/allData_imageNet_stop_epochs.csv"
        #datasetName = "../data/allData_imageNet_stop_epochs.csv"
        datasetName = "../data/allData_" + network + "_stop_epochs.csv"
        
        #datasetName = "allData_imageNet_stop_epochs.csv"
        #datasetName = "allData_" + network + "_stop_epochs.csv"
        self.data = self.readData(datasetName)


    def readData(self, filename):
        _data = dict()
        with open(filename, "r") as fl:
            for line in fl:
                if "alg" in line: continue # 1st line
#alg,epochs,lr,momentum,batch,lrAdv,momentumAdv,batchAdv,ratio,ratioAdv,epsilon,numIt,alpha,epsilonTest,numItTest,alphaTest,trainingError,testingError,advError,trainingTime,testingTime,adversarialTestingTime,stdTrainingError,stdTestingError,stdAdvError,stdTrainingTime,stdTestingTime,stdAdversarialTestingTime
                aux = line.split(",")

                alg = aux[0]

                epoch = int(aux[1])
                lr = float(aux[2])
                momentum = float(aux[3])
                bacth = int(aux[4])

                lrAdv = float(aux[5])
                momentumAdv = float(aux[6])
                bacthAdv = int(aux[7])

                ratio = float(aux[8])
                ratioAdv = float(aux[9])

                epsilon = float(aux[10])
                numIt = int(aux[11])
                alpha = float(aux[12])

                epsilonTest = float(aux[13])
                numItTest = int(aux[14])
                alphaTest = float(aux[15])

                trainError = float(aux[16])
                testError = float(aux[17])
                advError = float(aux[18])
                trainTime = float(aux[19])

                config = {'alg': alg,
                    'epoch':  epoch,
                    'batch_size':  bacth,
                    'learning_rate':  lr,
                    'momentum': momentum, 
                    'batch_size_adv':  bacthAdv,
                    'learning_rate_adv':  lrAdv,
                    'momentum_adv': momentumAdv, 
                    'ratio': ratio, 
                    'ratioAdv': ratioAdv, 
                    'epsilon': epsilon, 
                    'numIt': numIt,
                    'alpha': alpha,
                    }

                config = self._sortConfigDict(config)

                key_result = str({'epsilonTest':epsilonTest, 'numItTest':numItTest, 'alphaTest':alphaTest})
                result = {'trainError': trainError, 'testError':testError, 'advError':advError, 'trainTime':trainTime}
                key = str(config)

                #a = "{'alg': 'fgsm', 'epoch': 1, 'batch_size': 256, 'learning_rate': 0.1, 'momentum': 0.9, 'batch_size_adv': 256, 'learning_rate_adv': 0.1, 'momentum_adv': 0.9, 'ratio': 0.5, 'ratioAdv': 0.7, 'epsilon': 2.0, 'numIt': 20, 'alpha': 0.01}"
                #if key == a:    
                #    print(key)

                if key not in _data.keys(): _data[key] = dict()
                
                _data[key][key_result] = result
        return _data


    def getVal(self, config, epoch, adv_method):

        alpha = config['alpha']
        if adv_method == 'pgd_5':
            _adv_method = 'pgd'
            numIt=5
        elif adv_method == 'pgd_10':
            _adv_method = 'pgd'
            numIt=10
        elif adv_method == 'pgd_20':
            _adv_method = 'pgd'
            numIt=20
        else: #fsgm
            _adv_method = 'fgsm'
            numIt=0
            alpha = 0.0
        

        if config['ratio'] == 0 or config['ratioAdv'] == 0:
            #standard
            _adv_method = "standard"
            numIt=0
            alpha = 0.0


        config_with_budgets = {'alg': _adv_method,
            'epoch':  int(epoch),
            'batch_size':  int(config['batch_size']),
            'learning_rate':  config['learning_rate'],
            'momentum': config['momentum'], 
            'batch_size_adv':  int(config['batch_size_adv']),
            'learning_rate_adv':  config['learning_rate_adv'],
            'momentum_adv': config['momentum_adv'], 
            'ratio': config['ratio'], 
            'ratioAdv': config['ratioAdv'], 
            'epsilon': config['epsilon'], 
            'numIt': numIt,
            'alpha': alpha,
            }     

        config_with_budgets = self._sortConfigDict(config_with_budgets)

        key_result = str({'epsilonTest':Bound_epsilon, 'numItTest':20, 'alphaTest':0.01})
        result = self.data[str(config_with_budgets)][key_result]
        #result = {'trainError': trainError, 'testError':testError, 'advError':advError, 'trainTime':trainTime}
        return config_with_budgets, result


    def _checkConfig(self, config):
        if config['alg'] == "fgsm":
                # fgsm and standard
                config['alpha'] = 0.0                
                config['alg'] = "fgsm"
                            

        if config['ratio'] == 0 or config['ratioAdv'] == 0:
            #standard
            config['alpha'] = 0.0
            
            config['batch_size_adv'] = 0
            config['learning_rate_adv'] = 0.0
            config['momentum_adv'] = 0.0
            config['epsilon'] = 0.0

            config['ratio'] = 0.0
            config['ratioAdv'] = 0.0
        
        elif config['ratio'] == 1:
            #only adversarial
            config['batch_size'] = 0
            config['learning_rate'] = 0.0
            config['momentum'] = 0.0

        return self._sortConfigDict(config, skip=True)


    def _sortConfigDict(self, config, skip=False):
        myKeys = list(config.keys())
        myKeys.sort()
        config_order = {i: config[i] for i in myKeys}
        return config_order


    def config_to_vector(self, config):
        # convert a config dict into a numpy array to be used in the models
        _list_config = []
        for hp_name, val in config.items():
            if isinstance(val, str): # not numeric
                it = 0
                try:
                    HPvals = self.configspace.get_hyperparameter(hp_name).choices
                except:
                    HPvals = self.configspace.get_hyperparameter(hp_name).sequence

                for hp_val in HPvals:
                    if hp_val==val:
                        _list_config.append(it)
                        break
                    it += 1

            elif isinstance(val, list): # not numeric - budgetss
                for vv in val:
                    _list_config.append(vv)
            else:
                _list_config.append(val)

        return _list_config


    def compute(self, config):
        config = self._checkConfig(config)
        #print("config " + str(config))

        starttime = time.time()
        config_with_budgets, result = self.getVal(config, 16, config['alg']) # budget is epochs

        #result = {'trainError': trainError, 'testError':testError, 'advError':advError, 'trainTime':trainTime}
        trainTime = result['trainTime']
        loss_ = result['testError']
        advloss = result['advError']

        loss = self.a*loss_ + (1-self.a)*advloss
        #working_time = (trainTime)/(self.factor*1.0)
        #if working_time > 0:
        #    time.sleep(working_time)
        
        t_now = time.time()

        print("\nRunning config " + str(config_with_budgets) + " that achieved a accuracy of " + str(1-loss) + "  during " + str(t_now - starttime) + " seconds " + str((t_now-starttime)*self.factor))

        return loss, trainTime





def main(seed):
    if not os.path.isdir("./logs/"): os.mkdir("./logs/")
    if not os.path.isdir("./logs/run" + str(seed)): os.mkdir("./logs/run" + str(seed))
    file_logs = os.path.join("./logs/run" + str(seed), 'logs.csv')
    overwrite=True
    try:
        with open(file_logs, 'x') as fh: pass
    except FileExistsError:
        if overwrite:
            with open(file_logs, 'w') as fh: pass
        else:
            raise FileExistsError('The file %s already exists.'%file_logs)
    except:
        print("ERROR: logs already exist. please backup or delete file")
        raise
    
    with open(file_logs, 'a') as fh:
        fh.write("runID;it;incumbent;incTime;incAcc;incAccAdv;incAccFinal;budget;configTested;Time;Acc;AccAdv;AccFinal;StartTime;EndTime;RealTime;Overhead;incumbentRec;incTimeRec;incAccRec;incAccAdvRec;incAccFinalRec\n")

    configspace = get_configspace(seed)
    work = model_benchmark(configspace)

    if acqFunc == "random":
        random(configspace, seed, work, file_logs)
    else:
        BO(configspace, seed, work, file_logs)


def random(configspace, seed, work, file_logs):

    inc = 0
    incAcc = 0
    incTime = 0
    configs = configspace.sample_configuration(5) + configspace.sample_configuration(95)
    for it, conf_ in enumerate(configs):
        conf = dict(conf_)
        init = time.time()
        loss, trainTime = work.compute(conf)
        acc = 1-loss

        overhead = time.time()-init
        if acc > incAcc:
            inc = conf
            incAcc = acc
            incTime = trainTime 

        write_logs(file_logs, seed, it, conf, 16, trainTime, acc, inc, incTime, incAcc, overhead)


def BO(configspace, seed, work, file_logs):
    configs = configspace.sample_configuration(5)
    
    inc = 0
    incAcc = 0
    incTime = 0
    list_tested_x = []
    list_tested_y = []
    for it, conf_ in enumerate(configs):
        conf = dict(conf_)
        init = time.time()
        loss, trainTime = work.compute(conf)
        acc = 1-loss

        overhead = time.time()-init
        if acc > incAcc:
            inc = conf
            incAcc = acc
            incTime = trainTime 


        list_tested_x.append(work.config_to_vector(work._checkConfig(conf) ))
        list_tested_y.append(acc)

        write_logs(file_logs, seed, it, conf, 16, trainTime, acc, inc, incTime, incAcc, overhead)

    train_x = torch.tensor(list_tested_x, **tkwargs)
    train_obj = torch.tensor(list_tested_y, **tkwargs).unsqueeze(-1)


    for it in range(5, 101):
        init = time.time()

        model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=1),)
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        configs_test = configspace.sample_configuration(100)
        best_conf = None
        best_ei = -1

        #https://botorch.org/tutorials/compare_mc_analytic_acquisition
        model.eval()
        EI = ExpectedImprovement(model, best_f=incAcc)

        for config in configs_test:
            config = dict(config)
            x = torch.tensor([work.config_to_vector(work._checkConfig(config) )], **tkwargs)
            ei = EI(x)
            if best_conf is None or ei > best_ei:
                best_conf = config
                best_ei = ei

        loss, trainTime = work.compute(best_conf)
        acc = 1-loss

        overhead = time.time()-init
        if acc > incAcc:
            inc = best_conf
            incAcc = acc
            incTime = trainTime 

        conf_vector = work.config_to_vector(work._checkConfig(best_conf) )
        list_tested_x.append(conf_vector)
        list_tested_y.append(acc)

        conf_vector = torch.tensor([conf_vector], **tkwargs)
        acc = torch.tensor([acc], **tkwargs).unsqueeze(-1)
        train_x = torch.cat([train_x, conf_vector])
        train_obj = torch.cat([train_obj, acc])


        write_logs(file_logs, seed, it, best_conf, 16, trainTime, acc, inc, incTime, incAcc, overhead)

         
def write_logs(file_logs, seed, it, config, budget, trainTime, acc, inc, incTime, incAcc, overhead): 


    strWrite = str(seed) + ";" + str(it) + ";" + str(inc) + ";" + str(incTime) + ";" + str(incAcc) + ";" + \
					str(budget) + ";" + str(config) + ";" + str(trainTime) + ";" + str(acc) + ";" + str(overhead) + "\n"
    
    with open(file_logs, 'a') as fh:
        fh.write(strWrite)


def get_configspace(seed):
            
            if network == 'imageNet':
                batch_size_list = [256, 512]
                momentums_list = [0.9, 0.99]
            elif network == 'cifar10':
                momentums_list = [0.0, 0.9]
                batch_size_list = [100, 256]
            elif network == 'svhn':
                batch_size_list = [100, 256]
                momentums_list = [0.0, 0.9]
            else: #if BENCHMARK == 'mnist':
                batch_size_list = [128, 256]
                momentums_list = [0.9, 0.99]

            cs = CS.ConfigurationSpace(seed=seed)
            #cs = _ConfigSpace_(seed=seed)
            batch_sizes = CSH.OrdinalHyperparameter('batch_size', batch_size_list)
            learning_rates = CSH.OrdinalHyperparameter('learning_rate', [0.1, 0.01])
            momentums = CSH.OrdinalHyperparameter('momentum', momentums_list)
            batch_sizes_adv = CSH.OrdinalHyperparameter('batch_size_adv', batch_size_list)
            learning_rates_adv = CSH.OrdinalHyperparameter('learning_rate_adv', [0.1, 0.01])
            momentums_adv = CSH.OrdinalHyperparameter('momentum_adv', momentums_list)
            ratios = CSH.OrdinalHyperparameter('ratio', [0, 0.3, 0.5, 0.7, 1.0])
            ratioAdvs = CSH.OrdinalHyperparameter('ratioAdv', [0.3, 0.5, 0.7, 1.0])
            epsilons = CSH.OrdinalHyperparameter('epsilon', [Bound_epsilon])
            #numIts = CSH.OrdinalHyperparameter('numIt', [20]) #[5, 10, 20])
            alphas = CSH.OrdinalHyperparameter('alpha', [0.01])
            
            #adv_alg = CSH.CategoricalHyperparameter('alg', ['fgsm', 'pgd_5', 'pgd_10', 'pgd_20'])
            adv_alg = CSH.CategoricalHyperparameter('alg', ['pgd_20']) # pgd is the full budget

            list_hp = [batch_sizes, learning_rates, momentums, batch_sizes_adv, learning_rates_adv, momentums_adv,\
                        ratios, ratioAdvs, epsilons, alphas, adv_alg]

            cs.add_hyperparameters(list_hp)

            #conditions/constraints in the search space

            # if only adversarial training
            cond_ratio1 = OrConjunction(EqualsCondition(cs['batch_size_adv'], cs['ratio'], 1), \
                                        AndConjunction( GreaterThanCondition(cs['batch_size_adv'],cs['ratio'], 0), \
                                                        LessThanCondition(cs['batch_size_adv'], cs['ratio'], 1)))
            cond_ratio2 = OrConjunction(EqualsCondition(cs['learning_rate_adv'], cs['ratio'], 1), \
                                        AndConjunction( GreaterThanCondition(cs['learning_rate_adv'],cs['ratio'], 0), \
                                                        LessThanCondition(cs['learning_rate_adv'], cs['ratio'], 1)))
            cond_ratio3 = OrConjunction(EqualsCondition(cs['momentum_adv'], cs['ratio'], 1), \
                                        AndConjunction( GreaterThanCondition(cs['momentum_adv'],cs['ratio'], 0), \
                                                        LessThanCondition(cs['momentum_adv'], cs['ratio'], 1)))
            cs.add_condition(cond_ratio1)
            cs.add_condition(cond_ratio2)
            cs.add_condition(cond_ratio3)


            cond_ratio1 = OrConjunction(EqualsCondition(cs['batch_size'], cs['ratio'], 0), \
                                        AndConjunction( GreaterThanCondition(cs['batch_size'],cs['ratio'], 0), \
                                                        LessThanCondition(cs['batch_size'], cs['ratio'], 1)))
            cond_ratio2 = OrConjunction(EqualsCondition(cs['learning_rate'], cs['ratio'], 0), \
                                        AndConjunction( GreaterThanCondition(cs['learning_rate'],cs['ratio'], 0), \
                                                        LessThanCondition(cs['learning_rate'], cs['ratio'], 1)))
            cond_ratio3 = OrConjunction(EqualsCondition(cs['momentum'], cs['ratio'], 0), \
                                        AndConjunction( GreaterThanCondition(cs['momentum'],cs['ratio'], 0), \
                                                        LessThanCondition(cs['momentum'], cs['ratio'], 1)))
            cs.add_condition(cond_ratio1)
            cs.add_condition(cond_ratio2)
            cs.add_condition(cond_ratio3)

            #standard training
            cs.add_condition(NotEqualsCondition(cs['epsilon'], cs['ratio'], 0)) 
            cs.add_condition(NotEqualsCondition(cs['ratioAdv'], cs['ratio'], 0)) 


            #cs.add_condition(NotEqualsCondition(cs['alpha'], cs['alg'], 'fgsm'))

            return cs


if __name__== "__main__":
    for i in range (1, 21):
        main(i)

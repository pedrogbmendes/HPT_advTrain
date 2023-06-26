import os, pickle, sys
import torch, time
import numpy as np

from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.test_functions.multi_fidelity import AugmentedHartmann, AugmentedRosenbrock
from botorch.models import SingleTaskMultiFidelityGP, SingleTaskGP
from botorch.models.cost import AffineFidelityCostModel
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from typing import Optional
from torch import Tensor

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import EqualsCondition, LessThanCondition, GreaterThanCondition, OrConjunction, NotEqualsCondition, AndConjunction




network = 'imageNet'
#network = 'cifar10'
#network = 'svhn'
#network = 'mnist'



budget_option = 0 # all budgets
budget_option = 1 # epochs as budget
#budget_option = 2  # pgd it


budget1 = [1, 2, 4, 8, 16] # epochs
budget2 = [1, 5, 10, 20] #pgd it - 1 is fgsm
#budget2 = [5, 10, 20] #pgd it - 1 is fgsm
#budget2 = [1, 20] #pgd it - 1 is fgsm




torch.set_printoptions(precision=7, sci_mode=True)
tkwargs = {
    "dtype": torch.double,
    #"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}
n=16 # number of configs for warmstarting 

SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
N_ITER = 100-n if not SMOKE_TEST else 2
BATCH_SIZE = 1


# config = {'alg': alg,
#     'epoch':  epoch,
#     'batch_size':  bacth,
#     'learning_rate':  lr,
#     'momentum': momentum, 
#     'batch_size_adv':  bacthAdv,
#     'learning_rate_adv':  lrAdv,
#     'momentum_adv': momentumAdv, 
#     'ratio': ratio, 
#     'ratioAdv': ratioAdv, 
#     'epsilon': epsilon, 
#     'numIt': numIt,
#     'alpha': alpha,
#     }

#alpha and epsilon are fixed in this case  
# epoch and numIt are a possible budget


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




aux = len(budget2)
if budget_option == 1:
    pgd5_val=-1
    pgd10_val=-1
    pgd20_val=0
else:
    if aux == 2: 
        pgd5_val=-1
        pgd10_val=-1
        pgd20_val=1
    elif aux == 3:
        pgd5_val=0
        pgd10_val=1
        pgd20_val=2
    else:
        pgd5_val=1
        pgd10_val=2
        pgd20_val=3


dim=12
if budget_option == 0:
    no_fidelities = 2 
    target_fidelities = {dim-2: budget1[-1], dim-1:budget2[-1]}
    #fidelity_weights = {dim-2: 1.0, dim-1: 1.0}
    columns=[dim-2, dim-1]
    values=[budget1[-1],budget2[-1]]
    value_fidelities = [budget1, budget2]

    val_fidelities = list()
    for v in value_fidelities[0]:
        for z in value_fidelities[1]:
            val_fidelities.append({dim-2: v, dim-1: z})

else:
    no_fidelities = 1
    #fidelity_weights = {dim-2: 1.0, dim-1: 1.0}
    columns=[dim-1]
    if budget_option == 1:  
        values=[budget1[-1]]
        value_fidelities = [budget1]
        target_fidelities = {dim-1:budget1[-1]}
    else:
        value_fidelities = [budget2]
        values=[budget2[-1]]
        target_fidelities = {dim-1:budget2[-1]}

    val_fidelities = list()
    for v in value_fidelities[0]:
        val_fidelities.append({dim-1: v})


fidelities_sizes = []
for i in value_fidelities:
    fidelities_sizes.append(len(i))
largerFid = max(fidelities_sizes)
for i in value_fidelities:
    while len(i) < largerFid:
        i.append(-1)

fidelities = torch.tensor(value_fidelities, **tkwargs)


class model_benchmark(SyntheticTestFunction):

    _optimal_value = 0.0

    def __init__(self, configspace, bounds, dim=3, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""
        Args:
            dim: The (input) dimension. Must be at least 3.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        if dim < 3:
            raise ValueError(
                "AugmentedRosenbrock must be defined it at least 3 dimensions"
            )
        self.dim = dim
        self._bounds = bounds #[(-5.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]


        self.a = 0.5
        self.factor = 10000.0
        self.configspace= configspace

        #datasetName = "./hpbandster/workers/data/allData_imageNet_stop_epochs.csv"
        #datasetName = "../data/allData_imageNet_stop_epochs.csv"
        datasetName = "../data/allData_" + network + "_stop_epochs.csv"

        #datasetName = "allData_imageNet_stop_epochs.csv"
        datasetName = "allData_" + network + "_stop_epochs.csv"

        self.data = self.readData(datasetName)
        self.trainingSet = dict()
        self.pauseResume = True

        super().__init__(noise_std=noise_std, negate=negate)


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
        if adv_method == 'pgd_5' or adv_method == pgd5_val:
            _adv_method = 'pgd'
            numIt=5
        elif adv_method == 'pgd_10' or adv_method == pgd10_val:
            _adv_method = 'pgd'
            numIt=10
        elif adv_method == 'pgd_20' or adv_method == pgd20_val:
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
        if budget_option == 1:# epochs as budget
            if config['alg'] == "pgd_5" or config['alg'] == pgd5_val:
                config['alg'] = "pgd_5"

            elif config['alg'] == "pgd_10" or config['alg'] == pgd10_val:
                config['alg'] = "pgd_10"
                
            elif config['alg'] == "pgd_20" or config['alg'] == pgd20_val:
                config['alg'] = "pgd_20"

            else:
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
        if budget_option == 0:
            config_order = {i: config[i] for i in myKeys if i!='epoch' and i!='alg'}
            if not skip:
                config_order['epoch'] = config['epoch']
                config_order['alg'] = config['alg']

        elif budget_option == 1:
            config_order = {i: config[i] for i in myKeys if i!='epoch'}
            if not skip:
                config_order['epoch'] = config['epoch']
        else:
            config_order = {i: config[i] for i in myKeys if i!='alg'}
            if not skip:
                config_order['alg'] = config['alg']

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


    def evaluate_true(self, X: Tensor, recommendation=False) -> Tensor:
        res = []

        for x in X:

            config_dict = dict()
            hp_name_list = self.configspace.get_hyperparameter_names()
            hp_name_list.sort()
            for i, hp_name in enumerate(hp_name_list):
                config_dict[hp_name] = x[i].item()
            config = self._checkConfig(config_dict)

            #print("config " + str(config))

            starttime = time.time()
            if budget_option==0:
                budget = [i.item() for i in x[-no_fidelities:]]
                if budget[1] == 1: _budget='fgsm'   
                elif budget[1] == 5: _budget='pgd_5' 
                elif budget[1] == 10: _budget='pgd_10'  
                else: _budget='pgd_20' 

                config_with_budgets, result, = self.getVal(config, budget[0], _budget) # all is epochs

            elif budget_option==1: 
                budget = x[-no_fidelities:].item()
                config_with_budgets, result = self.getVal(config, budget, 'pgd_20') # budget is epochs

            else: #if budget_option==2:
                budget = x[-no_fidelities:].item()
                if budget == 1: _budget='fgsm'   
                elif budget == 5: _budget='pgd_5' 
                elif budget == 10: _budget='pgd_10'  
                else: _budget='pgd_20'  

                config_with_budgets, result = self.getVal(config, config['epoch'], _budget) # budget is adv methods

            prev_time = 0
            prev_budget = 0 
            if self.pauseResume:
                for kk in self.trainingSet.keys():
                    conf = pickle.loads(kk)
                    if budget_option!=2: # all budgets -> epochs is tracable
                        if conf['alg'] == config_with_budgets['alg'] \
                            and conf['batch_size'] == config_with_budgets['batch_size'] \
                            and conf['learning_rate'] == config_with_budgets['learning_rate'] \
                            and conf['momentum'] == config_with_budgets['momentum'] \
                            and conf['batch_size_adv'] == config_with_budgets['batch_size_adv']\
                            and conf['learning_rate_adv'] == config_with_budgets['learning_rate_adv'] \
                            and conf['momentum_adv'] == config_with_budgets['momentum_adv'] \
                            and conf['ratio'] == config_with_budgets['ratio'] \
                            and conf['ratioAdv'] == config_with_budgets['ratioAdv'] \
                            and conf['epsilon'] == config_with_budgets['epsilon'] \
                            and conf['numIt'] == config_with_budgets['numIt'] \
                            and conf['alpha'] == config_with_budgets['alpha']:
                            
                            if conf['epoch'] > prev_budget: # restore the largest tested budget of that config
                                prev_time = self.trainingSet[kk][3]
                                prev_budget = conf['epoch']                      

                    #if budget_option==2: # budget is pgd iteration that isn't tracable


            #result = {'trainError': trainError, 'testError':testError, 'advError':advError, 'trainTime':trainTime}
            trainTime = result['trainTime']
            acc = 1 - result['testError']
            advAcc = 1 - result['advError']

            acc_final = self.a*acc + (1-self.a)*advAcc

            #working_time = (trainTime)/(self.factor*1.0)
            #working_time = (trainTime-prev_time)/(self.factor*1.0)
            #if working_time > 0:
            #    time.sleep(working_time)
    
            key = pickle.dumps(config_with_budgets)
            self.trainingSet[key] = [config_with_budgets, acc, advAcc, trainTime] # 

            t_now = time.time()
            #real_time =  (t_now-starttime)*self.factor
            real_time = trainTime-prev_time 
            if not recommendation:
                print("\nRunning config " + str(config_with_budgets) + " that achieved a acc of " + str(acc_final) + "  during " + str(t_now - starttime) \
                        + " seconds " + str(real_time))
            else:
                print("\nRecomending config " + str(config_with_budgets) + " that achieved a balance acc of " + str(acc_final) + \
                        " and a acc of " + str(acc) +   " and a advAcc of " + str(advAcc) + \
                       "  during " + str(t_now - starttime) + " seconds " + str(real_time))

            res.append([config_with_budgets, budget, trainTime, acc, advAcc, acc_final, starttime, t_now, real_time])

        return res #torch.tensor(res, **tkwargs)


class multiFidelity_KG():
    def __init__(self, seed=1000):
        self.seed = seed
        torch.manual_seed(seed)

        self.configspace = self.get_configspace(budget_option+10000)

        self.cumulative_cost = 0.0
        self.discreteSpace = True
        self.it = 0

        self.inc = None
        self.inc_rec = None

        self.incTime = 0
        self.incTime_rec = 0

        self.incAcc = 0
        self.incAcc_rec = 0

        self.incAccAdv = 0
        self.incAccAdv_rec = 0

        self.incAccFinal = 0
        self.incAccFinal_rec = 0

        self.overhead = 0
        self.overhead_init = 0

        if not os.path.isdir("./logs/"): os.mkdir("./logs/")
        if not os.path.isdir("./logs/run" + str(self.seed)): os.mkdir("./logs/run" + str(self.seed))
        self.file_logs = os.path.join("./logs/run" + str(self.seed), 'logs.csv')
        overwrite=True
        try:
            with open(self.file_logs, 'x') as fh: pass
        except FileExistsError:
            if overwrite:
                with open(self.file_logs, 'w') as fh: pass
            else:
                raise FileExistsError('The file %s already exists.'%self.file_logs)
        except:
            print("ERROR: logs already exist. please backup or delete file")
            raise
        
        with open(self.file_logs, 'a') as fh:
            fh.write("runID;it;incumbent;incTime;incAcc;incAccAdv;incAccFinal;budget;configTested;Time;Acc;AccAdv;AccFinal;StartTime;EndTime;RealTime;Overhead;incumbentRec;incTimeRec;incAccRec;incAccAdvRec;incAccFinalRec\n")


        if budget_option == 0:
            lower_fidelities = [budget1[0], budget2[0]]   
            upper_fidelities = [budget1[-1], budget2[-1]]   
        elif budget_option == 1:
            lower_fidelities = [budget1[0]]   
            upper_fidelities = [budget1[-1]]   
        else:
            lower_fidelities = [budget2[0]]   
            upper_fidelities = [budget2[-1]]   


        lower = []   
        upper = []
        self.hp_list = []
        hp_name_list = self.configspace.get_hyperparameter_names()
        hp_name_list.sort()
        for hp_name in hp_name_list:
            # if budget_option == 0 and (hp_name == 'epoch' or hp_name == 'numIt'): continue
            # if budget_option == 1 and (hp_name == 'epoch'): continue
            # if budget_option == 2 and (hp_name == 'numIt'): continue

            try:
                vals = self.configspace.get_hyperparameter(hp_name).choices
            except:
                vals = self.configspace.get_hyperparameter(hp_name).sequence

            if any(isinstance(val, str) for val in vals): # not numeric
                size_list = int(len(vals))
                vals = list(range(size_list))
            
            self.hp_list.append(list(vals))
            upper.append(max(vals))
            lower.append(min(vals))

        _bounds = [lower+lower_fidelities, upper+upper_fidelities]
        self.bounds = torch.tensor(_bounds, **tkwargs)
        self.problem = model_benchmark(self.configspace, _bounds)

        #self.cost_model = AffineFidelityCostModel(fidelity_weights=target_fidelities, fixed_cost=0.01)
        #self.cost_aware_utility = InverseCostWeightedUtility(cost_model=self.cost_model)

        self.train_x, self.train_obj, self.train_obj_cost, results = self.generate_initial_data(n=n)
        self.write_logs(results)
        

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]


    def get_nearest_valid_hp_array(self, x):
        nearest = []
        for j, hp_value in enumerate(x[:-no_fidelities]):
            # Approximate the random value to an allowed hp value
            near = self.find_nearest(self.hp_list[j], hp_value)
            nearest.append(near)
        
        if no_fidelities==2: nearest.append(x[-2])
        nearest.append(x[-1])

        return nearest


    # def toConfig(self, vector):
    #     if vector[0] == 1:
    #         alg = 'pgd'
    #     else:
    #         alg = 'fgsm'

    #     return {'alg': alg, 
    #         'alpha': vector[1], 
    #         'batch_size': vector[2], 
    #         'batch_size_adv': vector[3], 
    #         'epsilon': vector[4],
    #         'learning_rate': vector[5], 
    #         'learning_rate_adv': vector[6], 
    #         'momentum': vector[7], 
    #         'momentum_adv': vector[8], 
    #         'numIt': vector[9], 
    #         'ratio': vector[10], 
    #         'ratioAdv': vector[11], 
    #         'epoch': vector[12]}


    def run(self):
        for i in range(N_ITER):
            
            self.overhead_init = time.time()
            print("Iterarion number " + str(i))
            # train model
            model, mll = self.initialize_model(self.train_x, self.train_obj)
            fit_gpytorch_mll(mll)


            # train cost model
            cost_model, mll_cost  = self.initialize_model(self.train_x, self.train_obj_cost)
            fit_gpytorch_mll(mll_cost)
            self.cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

            # get acquisition function and optimize it 
            model.eval()
            cost_model.eval()

            mfkg_acqf = self.get_mfkg(model)
            new_x, new_obj, cost, results = self.optimize_mfkg_and_get_observation(mfkg_acqf)
            
            self.train_x = torch.cat([self.train_x, new_x])
            self.train_obj = torch.cat([self.train_obj, new_obj])
            self.train_obj_cost = torch.cat([self.train_obj_cost, cost])
            self.cumulative_cost += cost

            self.get_recommendation(model)
            self.write_logs(results)

            print(f"\ntotal cost: {self.cumulative_cost.item()}\n")


    def generate_initial_data(self, n=16):
        # generate training data
        if self.discreteSpace:
            l = []
            for c in self.configspace.sample_configuration(16):
                l.append(self.problem.config_to_vector(self.problem._checkConfig(dict(c)) ))
            train_x = torch.tensor(l, **tkwargs)
            
            for i in range(no_fidelities):
                train_f = fidelities[i][torch.randint(fidelities_sizes[i], (n, 1))]
                train_x = torch.cat((train_x, train_f), dim=1)

        else:
            train_x = torch.rand(n, dim, **tkwargs)

        results= self.problem(train_x) #.unsqueeze(-1) # add output dimension
        train_obj = torch.tensor([res[5] for res in results], **tkwargs).unsqueeze(-1) #[config_with_budgets, budget, trainTime, acc, advAcc, acc_final, starttime, t_now, real_time]
        train_obj_cost = torch.tensor([res[2] for res in results], **tkwargs).unsqueeze(-1) #[config_with_budgets, budget, trainTime, acc, advAcc, acc_final, starttime, t_now, real_time]
        
        #train_obj_cost = self.problem.evaluate_true(train_x, costReturn=True).unsqueeze(-1) # add output dimension
        return train_x, train_obj, train_obj_cost, results


    def initialize_model(self, train_x, train_obj):
        # define a surrogate model suited for a "training data"-like fidelity parameter
        # in dimensions dim-2 and dim-2
        #model = SingleTaskMultiFidelityGP(train_x, train_obj, outcome_transform=Standardize(m=1),)
        #model = SingleTaskMultiFidelityGP(train_x, train_obj, outcome_transform=Standardize(m=1), fidelity_features=train_x[:, -2:])
        model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=1),)
        model.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        return model, mll
        

    def project(self, X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)


    def get_mfkg(self, model):
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=dim,
            columns=columns,
            values=values,
        )

        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self.bounds[:,:-no_fidelities],
            q=1,
            num_restarts=10 if not SMOKE_TEST else 2,
            raw_samples=1024 if not SMOKE_TEST else 4,
            options={"batch_limit": 10, "maxiter": 200},
            )

        return qMultiFidelityKnowledgeGradient(
                model=model,
                num_fantasies=128 if not SMOKE_TEST else 2,
                current_value=current_value,
                cost_aware_utility=self.cost_aware_utility,
                project=self.project,
            )


    def optimize_mfkg_and_get_observation(self, mfkg_acqf):
        """Optimizes MFKG and returns a new candidate, observation, and cost."""
        if self.discreteSpace:
            #optimize_acqf_discrete
            candidates, _ = optimize_acqf_mixed(
                acq_function=mfkg_acqf,
                bounds=self.bounds,
                fixed_features_list=val_fidelities,
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                # batch_initial_conditions=X_init,
                options={"batch_limit": 5, "maxiter": 200},
            )
            # candidates, _ = optimize_acqf_discrete(
            #     acq_function=mfkg_acqf,
            #     choices=generate_initial_data(50),
            #     q=BATCH_SIZE,
            #     options={"batch_limit": 5, "maxiter": 200},
            # )  
            #       
        else:
            X_init = gen_one_shot_kg_initial_conditions(
                acq_function = mfkg_acqf,
                bounds=self.bounds,
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
            candidates, _ = optimize_acqf(
                acq_function=mfkg_acqf,
                bounds=self.bounds,
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                batch_initial_conditions=X_init,
                options={"batch_limit": 5, "maxiter": 200},
            )
        # observe new values
        #cost = self.cost_model(candidates).sum()
        new_x = candidates.detach()
        new_x = self.get_nearest_valid_hp_array(new_x.tolist()[0])
        new_x = torch.tensor([new_x])
        
        self.overhead = time.time()-self.overhead_init 

        results= self.problem(new_x) #.unsqueeze(-1) # add output dimension
        new_obj = torch.tensor([res[5] for res in results], **tkwargs).unsqueeze(-1) #[config_with_budgets, budget, trainTime, acc, advAcc, acc_final, starttime, t_now, real_time]
        cost = torch.tensor([res[2] for res in results], **tkwargs).unsqueeze(-1) #[config_with_budgets, budget, trainTime, acc, advAcc, acc_final, starttime, t_now, real_time]

        #cost = self.problem.evaluate_true(new_x, costReturn=True).unsqueeze(-1)

        #print(f"\ncandidates:\n{new_x}\n")
        #print(f"observations:\n{new_obj}\n\n")
        return new_x, new_obj, cost, results


    def get_recommendation(self, model):
        rec_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=dim,
            columns=columns,
            values=values,
        )

        final_rec, _ = optimize_acqf(
            acq_function=rec_acqf,
            bounds=self.bounds[:,:-no_fidelities],
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
        )
        
        final_rec = rec_acqf._construct_X_full(final_rec)
        final_rec = self.get_nearest_valid_hp_array(final_rec.tolist()[0])
        #final_rec_conf =  self.problem._checkConfig(self.toConfig(final_rec), complete=True)

        final_rec = torch.tensor([final_rec])
        #results = self.problem(final_rec) #.unsqueeze(-1) # add output dimension
        results = self.problem.evaluate_true(final_rec, recommendation=True) #.unsqueeze(-1) # add output dimension

        #[config_with_budgets, budget, trainTime, acc, advAcc, acc_final, starttime, t_now, real_time]
        self.inc_rec = results[0][0]
        self.incTime_rec = results[0][2]
        self.incAcc_rec = results[0][3]
        self.incAccAdv_rec = results[0][4]
        self.incAccFinal_rec = results[0][5]

        if self.incAccFinal_rec > self.incAccFinal:
            self.inc = self.inc_rec
            self.incTime = self.incTime_rec 
            self.incAcc = self.incAcc_rec
            self.incAccAdv = self.incAccAdv_rec
            self.incAccFinal = self.incAccFinal_rec

        return


    def get_configspace(self, seed):
            
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


            list_hp = [batch_sizes, learning_rates, momentums, batch_sizes_adv, learning_rates_adv, momentums_adv,\
                        ratios, ratioAdvs, epsilons, alphas]

            #if budget_option == 0: # all budgets
            #    batch_sizes = CSH.CategoricalHyperparameter('alg', ['fgsm', 'pgd'])
            #    epoch = CSH.CategoricalHyperparameter('epoch', [1,2,4,8,16])
            if budget_option == 1:# epochs as budget
                list_algs = ['fgsm', 'pgd_20']
                list_algs = ['pgd_20']
                #list_algs = ['pgd_5', 'pgd_10', 'pgd_20']
                #list_algs = ['fgsm', 'pgd_5', 'pgd_10', 'pgd_20']
                
                adv_alg = CSH.CategoricalHyperparameter('alg', list_algs)
                list_hp.append(adv_alg)

            elif budget_option == 2:
            #else: # pgd it
                epoch = CSH.CategoricalHyperparameter('epoch', [16])
                list_hp.append(epoch)


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


            #only pgd
            #if budget_option == 1:# epochs as budget
            #    #cs.add_condition(EqualsCondition(cs['numIt'], cs['alg'], 'pgd'))
            #    cs.add_condition(NotEqualsCondition(cs['alpha'], cs['alg'], 'fgsm'))

            return cs


    def write_logs(self, results): # config, budget, time, acc, starttime, fineltime, real_time):

        for config, budget, trainTime, acc, advAcc, acc_final, starttime, fineltime, real_time in results:
    
            strWrite = str(self.seed) + ";" + str(self.it) + ";" + \
                    str(self.inc) + ";" + str(self.incTime) + ";" + str(self.incAcc) + ";"  + str(self.incAccAdv) + ";"  + str(self.incAccFinal) + ";"  +  \
					str(budget) + ";" + str(config) + ";" + str(trainTime) + ";" + str(acc) + ";"  + str(advAcc) + ";"  + str(acc_final) + ";" + \
                    str(starttime) + ";" + str(fineltime) + ";" + str(real_time) + ";" + str(self.overhead) + ";" + \
                    str(self.inc_rec) + ";" + str(self.incTime_rec) + ";" + str(self.incAcc_rec) + ";" + str(self.incAccAdv_rec) + ";" + str(self.incAccFinal_rec) + "\n"
    
            with open(self.file_logs, 'a') as fh:
                fh.write(strWrite)
            self.it+=1


def main(it):
    for i in range(1,it+1):
        opt = multiFidelity_KG(i)
        opt.run()

if __name__ == "__main__":
    main(20)
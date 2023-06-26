import subprocess
import os, time
from concurrent.futures import ThreadPoolExecutor
import threading


#types = ['std', 'fgsm_rs']
types = ['fgsm', 'pgd']
types = ['pgd']

#datasets = ['cifar10']
#datasets = ['mnist']
datasets = ['svhn']
#datasets = ['imageNet']


lock = threading.Lock()
no_workers_per_job = 2
available_workers = [0,1]
no_jobs = 0


if len(datasets) != 1:
    print("ERROR: Use only one dataset at each time")

stopCondition = "epochs" 
#stopCondition = "time" 

if stopCondition == "time":   
    stop_vals = [600]
    if 'imageNet' in datasets:
        #stop_vals = [10000]
        stop_vals = [43200] #run for 12hour with 1 gpus
    elif 'svhn' in datasets:
        stop_vals = [21600]
else: #epochs
    stop_vals = [10]
    if 'imageNet' in datasets:
        stop_vals = [10]
    elif 'cifar10' in datasets:
        stop_vals = [20]
    elif 'svhn' in datasets:
        stop_vals = [15]


# ratio = [1.0]
# ratioAdv = [1.0]


ratio = [0.3, 0.5, 0.7, 1.0] # ratio/percentage of adversarial examples 
ratioAdv = [0.3, 0.5, 0.7, 1.0]

#alphas = [0.01, 0.001, 0.0001]
#alphas = [0.001, 0.0001]
alphas = [0.01]

#num_itearions = [5, 10]
num_itearions = [20]

# learning_rates = [0.1, 0.01]
# momentums = [0.9, 0.99]
# batchs = [256]

# learning_rates_advs = [0.1]
# momentums_advs = [0.0]
# batchs_advs = [100]


half_prec = False

learning_rates = [0.1, 0.01, 0.001]
momentums = [0.0, 0.9, 0.99]
batchs = [100, 256]

learning_rates_advs = [0.1, 0.01, 0.001]
momentums_advs = [0.0, 0.9, 0.99]
batchs_advs = [100, 256]

# learning_rates = [0.1]
# momentums = [0.0]
# batchs = [100]

# learning_rates_advs = [0.1]
# momentums_advs = [0.0]
# batchs_advs = [100]

if 'cifar10' in datasets:
    batchs = [100, 256]
    batchs_advs = [100, 256]

elif 'imageNet' in datasets:
    momentums = [0.9]
    momentums_advs = [0.9]
    batchs = [512]
    batchs_advs = [512]
    half_prec = True


def run(command, available_workers):

    workers = []
    if no_workers_per_job == 1:

        lock.acquire()
        #global available_workers
        workers_visible = available_workers.pop(0)
        lock.release()
        
        command += " --workers=" + str(workers_visible)
        workers.append(workers_visible)

        command = 'CUDA_VISIBLE_DEVICES=' + str(workers_visible) + ' ' + command

    else:
        while len(available_workers) > no_workers_per_job:
            #not enough workers
            print("Not enough workers")
            time.sleep(100)

        workers_visible = ''
        for i in range(no_workers_per_job):
            lock.acquire()
            worker = available_workers.pop(0) #global available_workers
            lock.release()

            workers_visible += str(worker) + "," 
            workers.append(worker)

        
        command = 'CUDA_VISIBLE_DEVICES=' + workers_visible[:-1] + ' ' + command + " --workers=" + workers_visible[:-1]
        
    print("Running: " + command + " on workers " + str(workers_visible))

    #time.sleep(worker)
    subprocess.run(command, shell=True)

    for w in workers:
        lock.acquire()
        #global available_workers
        available_workers.append(w)
        lock.release()

    global no_jobs
    lock.acquire()
    no_jobs -= 1
    lock.release()
    print("Job Done: " + command)


def submit_job(command, available_workers, executor):
    #if not available_workers:
    #    print("no idle workers.")
        #time.sleep(60) #sleep for 1 minute

    t =  executor.submit(run, command, available_workers) # run one 
    #print(t.result())


def main():
    executor = ThreadPoolExecutor(max_workers=int(len(available_workers)/no_workers_per_job))
    for _type in types:
        if _type == 'std':
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    for lr in learning_rates:
                        for mm in momentums:
                            for bs in batchs:
                                if lr==0.1 and mm==0.0 and bs==100: continue
                                command = "python3 train.py --type=std --dataset=" + _dataset + " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val)
                                command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                command += " --lr_adv=0 --momentum_adv=0 --batch_adv=0" 
                                command += " --half_prec=" + str(half_prec)
                                
                                submit_job(command, available_workers, executor)


        elif "pgd" in _type: # == 'robust pgd_rs' or _type == 'robust pgd':
        #robust pgd  adversarial  
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    for _ratio in ratio:
                        for _ratio_adv in ratioAdv:
                            if _dataset == 'cifar10':
                                bounds = [4, 8, 12, 16]
                            elif _dataset == 'mnist':
                                bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                            elif _dataset == 'imageNet':
                                bounds = [2, 4]
                            elif _dataset == 'svhn':
                                bounds = [4, 8, 12]
                            else:
                                print("wrong dataset")
                                continue

                            for _epsilon in bounds:
                                for _num_iter in num_itearions:
                                    for _alpha in alphas:
                                         for lr in learning_rates:
                                            for mm in momentums:
                                                for bs in batchs:
                                                    for lr_adv in learning_rates_advs:
                                                        for mm_adv in momentums_advs:
                                                            for bs_adv in batchs_advs:
                                                                if lr==0.1 and mm==0.0 and bs==100 and  lr_adv==0.1 and mm_adv==0.0 and bs_adv==100: continue

                                                                if _type == 'pgd':
                                                                    command = "python3 train.py --type=robust --alg=pgd " 
                                                                elif _type == 'pgd_rs':
                                                                    command = "python3 train.py --type=robust --alg=pgd_rs " 
                                                                else:
                                                                    print("ERROR in type")
                                                                    #command = "python3 train.py --type=both --alg=pgd " 
                                                                    #command += " --ratio_adv=" + str(_ratio_adv) 

                                                                command += " --dataset=" + _dataset 
                                                                command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                                command += " --ratio=" + str(_ratio) 
                                                                command += " --ratio_adv=" + str(_ratio_adv) 

                                                                command += " --epsilon=" + str(_epsilon) 
                                                                command += " --num_iter=" + str(_num_iter) 
                                                                command += " --alpha=" + str(_alpha) 
                                                                
                                                                command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                                                command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) #*no_workers_per_job)
                                                                command += " --half_prec=" + str(half_prec)
                                                                
                                                                submit_job(command, available_workers, executor)



        elif  'fgsm' in _type: #== '= fgsm_rs' or _type == 'robust fgsm' or _type == 'robust fgsm_free' or _type == 'robust fgsm_grad_align':
        #fgsm just adversarial  
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    for _ratio in ratio:
                        for _ratio_adv in ratioAdv:
                            if _dataset == 'cifar10':
                                bounds = [4, 8, 12, 16]
                            elif _dataset == 'mnist':
                                bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                            elif _dataset == 'imageNet':
                                bounds = [2, 4]
                            elif _dataset == 'svhn':
                                bounds = [4, 8, 12]
                            else:
                                print("wrong dataset")
                                continue

                            for _epsilon in bounds:
                                for lr in learning_rates:
                                    for mm in momentums:
                                        for bs in batchs:
                                            for lr_adv in learning_rates_advs:
                                                for mm_adv in momentums_advs:
                                                    for bs_adv in batchs_advs:
                                                        if lr==0.1 and mm==0.0 and bs==100 and lr_adv==0.1 and mm_adv==0.0 and bs_adv==100: continue

                                                        if _type == 'fgsm':
                                                            command = "python3 train.py --type=robust --alg=fgsm " 
                                                        elif _type ==  'fgsm_rs':
                                                            command = "python3 train.py --type=robust --alg=fgsm_rs " 
                                                        elif _type ==  'fgsm_free':
                                                            command = "python3 train.py --type=robust --alg=fgsm_free " 
                                                        elif _type ==  'fgsm_grad_align':
                                                            command = "python3 train.py --type=robust --alg=fgsm_grad_align " 
                                                        else:
                                                            print("ERROR in type")

                                                            #command = "python3 train.py --type=both --alg=fgsm "
                                                            #command += " --ratio_adv=" + str(_ratio_adv) 

                                                        command += " --dataset=" + _dataset 
                                                        command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                        command += " --ratio=" + str(_ratio) 
                                                        command += " --ratio_adv=" + str(_ratio_adv) 
                                                        
                                                        command += " --epsilon=" + str(_epsilon) 

                                                        command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                                        command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) # *no_workers_per_job)
                                                        command += " --half_prec=" + str(half_prec)
                                                       
                                                        submit_job(command, available_workers, executor)

        else:
            print("wronh type of experiment")



def main_preTrained():
    executor = ThreadPoolExecutor(max_workers=int(len(available_workers)/no_workers_per_job))
    global no_jobs

    for _type in types:
        if _type == 'std':
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    for lr in learning_rates:
                        for mm in momentums:
                            for bs in batchs:

                                command = "python3 train.py --type=std --dataset=" + _dataset + " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val)
                                command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                command += " --lr_adv=0 --momentum_adv=0 --batch_adv=0" 
                                command += " --half_prec=" + str(half_prec) + " --preTrain=1"
                                
                                lock.acquire()
                                no_jobs += 1
                                lock.release()

                                submit_job(command, available_workers, executor)


        elif "pgd" in _type: # == 'robust pgd_rs' or _type == 'robust pgd':
        #robust pgd  adversarial  
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    if any(item < 1.0 for item in ratio):
                        # lets train first standard and save the models
                        for lr in learning_rates:
                            for mm in momentums:
                                for bs in batchs:
                                    command = "python3 train.py --type=std --dataset=" + _dataset + " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val)
                                    command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                    command += " --lr_adv=0 --momentum_adv=0 --batch_adv=0" 
                                    command += " --half_prec=" + str(half_prec) + " --preTrain=2"
                                    submit_job(command, available_workers, executor)

                                    lock.acquire()
                                    no_jobs += 1
                                    lock.release()

                    while no_jobs > 0 : 
                        time.sleep(5)

                    for _ratio in ratio:
                        for _ratio_adv in ratioAdv:
                            if _dataset == 'cifar10':
                                bounds = [4, 8, 12, 16]
                            elif _dataset == 'mnist':
                                bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                            elif _dataset == 'imageNet':
                                bounds = [2, 4]
                            elif _dataset == 'svhn':
                                bounds = [4, 8, 12]
                            else:
                                print("wrong dataset")
                                continue

                            for _epsilon in bounds:
                                for _num_iter in num_itearions:
                                    for _alpha in alphas:
                                        for lr_adv in learning_rates_advs:
                                            for mm_adv in momentums_advs:
                                                for bs_adv in batchs_advs:
                                                    if _ratio==1.0: # only adversarial

                                                        if _type == 'pgd':
                                                            command = "python3 train.py --type=robust --alg=pgd " 
                                                        elif _type == 'pgd_rs':
                                                            command = "python3 train.py --type=robust --alg=pgd_rs " 
                                                        else:
                                                            print("ERROR in type")

                                                        command += " --dataset=" + _dataset 
                                                        command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                        command += " --ratio=" + str(_ratio) 
                                                        command += " --ratio_adv=" + str(_ratio_adv) 

                                                        command += " --epsilon=" + str(_epsilon) 
                                                        command += " --num_iter=" + str(_num_iter) 
                                                        command += " --alpha=" + str(_alpha) 
                                                                
                                                        command += " --lr=0 --momentum=0 --batch=0"
                                                        command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) #*no_workers_per_job)
                                                        command += " --half_prec=" + str(half_prec) + " --preTrain=2"
                                                                
                                                        lock.acquire()
                                                        no_jobs += 1
                                                        lock.release()

                                                        submit_job(command, available_workers, executor)

                                                    else:
                                                        for lr in learning_rates:
                                                            for mm in momentums:
                                                                for bs in batchs:

                                                                    if _type == 'pgd':
                                                                        command = "python3 train.py --type=robust --alg=pgd " 
                                                                    elif _type == 'pgd_rs':
                                                                        command = "python3 train.py --type=robust --alg=pgd_rs " 
                                                                    else:
                                                                        print("ERROR in type")
                                                                        #command = "python3 train.py --type=both --alg=pgd " 
                                                                        #command += " --ratio_adv=" + str(_ratio_adv) 

                                                                    command += " --dataset=" + _dataset 
                                                                    command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                                    command += " --ratio=" + str(_ratio) 
                                                                    command += " --ratio_adv=" + str(_ratio_adv) 

                                                                    command += " --epsilon=" + str(_epsilon) 
                                                                    command += " --num_iter=" + str(_num_iter) 
                                                                    command += " --alpha=" + str(_alpha) 
                                                                    
                                                                    command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) 
                                                                    command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) #*no_workers_per_job)
                                                                    command += " --half_prec=" + str(half_prec) + " --preTrain=2"
                                                                    
                                                                    lock.acquire()
                                                                    no_jobs += 1
                                                                    lock.release()

                                                                    submit_job(command, available_workers, executor)



        elif  'fgsm' in _type:
        #fgsm just adversarial  
            for _dataset in datasets: 
                for _stop_val in stop_vals:
                    if any(item < 1.0 for item in ratio):
                        # lets train first standard and save the models
                        for lr in learning_rates:
                            for mm in momentums:
                                for bs in batchs:
                                    command = "python3 train.py --type=std --dataset=" + _dataset + " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val)
                                    command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                    command += " --lr_adv=0 --momentum_adv=0 --batch_adv=0" 
                                    command += " --half_prec=" + str(half_prec) + " --preTrain=2"
                                    submit_job(command, available_workers, executor)

                                    lock.acquire()
                                    no_jobs += 1
                                    lock.release()

                    while no_jobs > 0 : 
                        time.sleep(5)

                    for _ratio in ratio:
                        for _ratio_adv in ratioAdv:
                            if _dataset == 'cifar10':
                                bounds = [4, 8, 12, 16]
                            elif _dataset == 'mnist':
                                bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                            elif _dataset == 'imageNet':
                                bounds = [2, 4]
                            elif _dataset == 'svhn':
                                bounds = [4, 8, 12]
                            else:
                                print("wrong dataset")
                                continue

                            for _epsilon in bounds:
                                for lr_adv in learning_rates_advs:
                                    for mm_adv in momentums_advs:
                                        for bs_adv in batchs_advs:
                                            if _ratio==1.0:
                                                if _type == 'fgsm':
                                                    command = "python3 train.py --type=robust --alg=fgsm " 
                                                elif _type ==  'fgsm_rs':
                                                    command = "python3 train.py --type=robust --alg=fgsm_rs " 
                                                elif _type ==  'fgsm_free':
                                                    command = "python3 train.py --type=robust --alg=fgsm_free " 
                                                elif _type ==  'fgsm_grad_align':
                                                    command = "python3 train.py --type=robust --alg=fgsm_grad_align " 
                                                else:
                                                    print("ERROR in type")

                                                command += " --dataset=" + _dataset 
                                                command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                command += " --ratio=" + str(_ratio) 
                                                command += " --ratio_adv=" + str(_ratio_adv) 
                                                
                                                command += " --epsilon=" + str(_epsilon) 

                                                command += " --lr=0 --momentum=0 --batch=0"
                                                command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) # *no_workers_per_job)
                                                command += " --half_prec=" + str(half_prec) + " --preTrain=2"
                                            
                                                lock.acquire()
                                                no_jobs += 1
                                                lock.release()

                                                submit_job(command, available_workers, executor)

                                            else:
                                                for lr in learning_rates:
                                                    for mm in momentums:
                                                        for bs in batchs:

                                                            if _type == 'fgsm':
                                                                command = "python3 train.py --type=robust --alg=fgsm " 
                                                            elif _type ==  'fgsm_rs':
                                                                command = "python3 train.py --type=robust --alg=fgsm_rs " 
                                                            elif _type ==  'fgsm_free':
                                                                command = "python3 train.py --type=robust --alg=fgsm_free " 
                                                            elif _type ==  'fgsm_grad_align':
                                                                command = "python3 train.py --type=robust --alg=fgsm_grad_align " 
                                                            else:
                                                                print("ERROR in type")

                                                            command += " --dataset=" + _dataset 
                                                            command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 

                                                            command += " --ratio=" + str(_ratio) 
                                                            command += " --ratio_adv=" + str(_ratio_adv) 
                                                            
                                                            command += " --epsilon=" + str(_epsilon) 

                                                            command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs) #*no_workers_per_job)
                                                            command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv) # *no_workers_per_job)
                                                            command += " --half_prec=" + str(half_prec) + " --preTrain=2"
                                                        
                                                            lock.acquire()
                                                            no_jobs += 1
                                                            lock.release()

                                                            submit_job(command, available_workers, executor)

        else:
            print("wronh type of experiment")






if __name__ == '__main__':
    #main()
    main_preTrained()




# import subprocess
# import os, time
# from concurrent.futures import ThreadPoolExecutor
# import threading


# #types = ['std', 'robust pgd', 'robust pgd_rs', 'robust fgsm', 'robust fgsm_rs','robust fgsm_free', 'robust fgsm_grad_align', 'both pgd', 'both fgsm']
# #types = ['robust pgd', 'robust pgd_rs', 'robust fgsm', 'robust fgsm_rs','robust fgsm_free', 'robust fgsm_grad_align']
# types = ['robust pgd', 'robust fgsm']
# #types = ['robust pgd_rs', 'robust fgsm_rs']
# #types = ['both fgsm', 'both pgd']
# #types = ['both pgd']
# #types = ['robust fgsm_grad_align']

# #datasets = ['cifar10']
# #datasets = ['mnist']
# datasets = ['svhn']
# #datasets = ['imageNet']

# if len(datasets) != 1:
#     print("ERROR: Ru only one dataset at each time")

# #stopCondition = "epochs" 
# stopCondition = "time" 

# if stopCondition == "time":   
#     stop_vals = [600]
#     if 'imageNet' in datasets:
#         stop_vals = [3600]
#         stop_vals = [600]
#     elif 'svhn' in datasets:
#         stop_vals = [6*3600]
# else: #epochs
#     stop_vals = [10]
#     if 'imageNet' in datasets:
#         stop_vals = [50]

# #ratio = [0.7]
# ratio = [1.0]
# ratioAdv = [1.0]

# #ratio = [0.3, 0.5, 0.7] # ratio/percentage of adversarial examples 
# #ratioAdv = [0.3, 0.5, 0.7]

# #alphas = [0.01, 0.001, 0.0001]
# #alphas = [0.001, 0.0001]
# alphas = [0.01]

# #num_itearions = [5, 10, 20]
# num_itearions = [20]

# lock = threading.Lock()
# available_workers = [0,1]

# learning_rates = [0.1]
# momentums = [0.0]
# batchs = [100]

# learning_rates_advs = [0.1]
# momentums_advs = [0.0]
# batchs_advs = [100]


# # learning_rates = [0.1, 0.01, 0.001]
# # momentums = [0.0, 0.9, 0.99]
# # batchs = [100, 10, 1000]

# # learning_rates_advs = [0.1, 0.01, 0.001]
# # momentums_advs = [0.0, 0.9, 0.99]
# # batchs_advs = [100, 10, 1000]

# def run(command, available_workers):

#     lock.acquire()
#     #global available_workers
#     worker = available_workers.pop(0)
#     lock.release()

#     command += " --worker=" + str(worker)
#     print("Running: " + command + " on worker " + str(worker))

#     #time.sleep(worker)
#     subprocess.run(command, shell=True)

#     lock.acquire()
#     #global available_workers
#     available_workers.append(worker)
#     lock.release()
#     print("Job Done: " + command)



# def submit_job(command, available_workers, executor):
#     #if not available_workers:
#     #    print("no idle workers.")
#         #time.sleep(60) #sleep for 1 minute

#     t =  executor.submit(run, command, available_workers) # run one 
#     #print(t.result())


# def main():
#     counter = 0

#     executor = ThreadPoolExecutor(max_workers=int(len(available_workers)))
#     for _type in types:
#         auxratioAdv = ratioAdv
#         if "both" not in _type:
#             # ratio_adv is just used whrn the have both (ie first we perform standard training with clean data and then trin with adversarial examples)
#             auxratioAdv = [1]

#         if _type == 'std':
#             for _dataset in datasets: 
#                 for _stop_val in stop_vals:
#                     for lr in learning_rates:
#                         for mm in momentums:
#                             for bs in batchs:
#                                 command = "python3 train.py --type=std --dataset=" + _dataset + " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val)
#                                 command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs)
#                                 submit_job(command, available_workers, executor)


#         elif _type == 'robust pgd_rs' or _type == 'robust pgd':
#         #robust pgd just adversarial  
#             for _dataset in datasets: 
#                 for _stop_val in stop_vals:
#                     for _ratio in ratio:
#                         for _ratio_adv in auxratioAdv:
#                             if _dataset == 'cifar10':
#                                 bounds = [4, 8, 12, 16]
#                             elif _dataset == 'mnist':
#                                 bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#                             elif _dataset == 'imageNet':
#                                 bounds = [2, 4, 8]
#                                 bounds = [2]
#                             elif _dataset == 'svhn':
#                                 bounds = [4, 8, 12, 16]

#                             else:
#                                 print("wrong dataset")
#                                 continue

#                             for _epsilon in bounds:
#                                 for _num_iter in num_itearions:
#                                     for _alpha in alphas:
#                                          for lr in learning_rates:
#                                             for mm in momentums:
#                                                 for bs in batchs:
#                                                     if _type == 'robust pgd':
#                                                         command = "python3 train.py --type=robust --alg=pgd " 
#                                                     elif _type == 'robust pgd_rs':
#                                                         command = "python3 train.py --type=robust --alg=pgd_rs " 
#                                                     else:
#                                                         print("ERROR in type")
#                                                         #command = "python3 train.py --type=both --alg=pgd " 
#                                                         #command += " --ratio_adv=" + str(_ratio_adv) 

#                                                     command += " --dataset=" + _dataset 
#                                                     command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 
#                                                     command += " --ratio=" + str(_ratio) 
#                                                     command += " --epsilon=" + str(_epsilon) 
#                                                     command += " --num_iter=" + str(_num_iter) 
#                                                     command += " --alpha=" + str(_alpha) 
                                                    
#                                                     command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs)
#                                                     submit_job(command, available_workers, executor)

#         elif  _type == 'both pgd':
#         #standard + pgd
#             for _dataset in datasets: 
#                 for _stop_val in stop_vals:
#                     for _ratio in ratio:
#                         for _ratio_adv in auxratioAdv:
#                             if _dataset == 'cifar10':
#                                 bounds = [4, 8, 12, 16]
#                             elif _dataset == 'mnist':
#                                 bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#                             elif _dataset == 'imageNet':
#                                 bounds = [2, 4, 8]
#                                 bounds = [2]
#                             elif _dataset == 'svhn':
#                                 bounds = [4, 8, 12, 16]
#                             else:
#                                 print("wrong dataset")
#                                 continue

#                             for _epsilon in bounds:
#                                 for _num_iter in num_itearions:
#                                     for _alpha in alphas:
#                                          for lr in learning_rates:
#                                             for mm in momentums:
#                                                 for bs in batchs:
#                                                     for lr_adv in learning_rates_advs:
#                                                         for mm_adv in momentums_advs:
#                                                             for bs_adv in batchs_advs:
#                                                                 command = "python3 train.py --type=both --alg=pgd " 

#                                                                 command += " --dataset=" + _dataset 
#                                                                 command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 
#                                                                 command += " --ratio=" + str(_ratio) 
#                                                                 command += " --ratio_adv=" + str(_ratio_adv) 

#                                                                 command += " --epsilon=" + str(_epsilon) 
#                                                                 command += " --num_iter=" + str(_num_iter) 
#                                                                 command += " --alpha=" + str(_alpha) 
                                                                
#                                                                 command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs)
#                                                                 command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv)

#                                                                 counter += 1
#                                                                 print("submiting job number " + str(counter))
#                                                                 submit_job(command, available_workers, executor)

#         elif  _type == 'robust fgsm_rs' or _type == 'robust fgsm' or _type == 'robust fgsm_free' or _type == 'robust fgsm_grad_align':
#         #fgsm just adversarial  
#             for _dataset in datasets: 
#                 for _stop_val in stop_vals:
#                     for _ratio in ratio:
#                         for _ratio_adv in auxratioAdv:
#                             if _dataset == 'cifar10':
#                                 bounds = [4, 8, 12, 16]
#                             elif _dataset == 'mnist':
#                                 bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#                             elif _dataset == 'imageNet':
#                                 bounds = [2, 4, 8]
#                                 bounds = [2]
#                             elif _dataset == 'svhn':
#                                 bounds = [4, 8, 12, 16]
#                             else:
#                                 print("wrong dataset")
#                                 continue

#                             for _epsilon in bounds:
#                                 for lr in learning_rates:
#                                     for mm in momentums:
#                                         for bs in batchs:
#                                             if _type == 'robust fgsm':
#                                                 command = "python3 train.py --type=robust --alg=fgsm " 
#                                             elif _type ==  'robust fgsm_rs':
#                                                 command = "python3 train.py --type=robust --alg=fgsm_rs " 
#                                             elif _type ==  'robust fgsm_free':
#                                                 command = "python3 train.py --type=robust --alg=fgsm_free " 
#                                             elif _type ==  'robust fgsm_grad_align':
#                                                 command = "python3 train.py --type=robust --alg=fgsm_grad_align " 
#                                             else:
#                                                 print("ERROR in type")

#                                                 #command = "python3 train.py --type=both --alg=fgsm "
#                                                 #command += " --ratio_adv=" + str(_ratio_adv) 

#                                             command += " --dataset=" + _dataset 
#                                             command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 
#                                             command += " --ratio=" + str(_ratio) 
#                                             command += " --epsilon=" + str(_epsilon) 

#                                             command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs)
#                                             submit_job(command, available_workers, executor)

#         elif _type =='both fgsm':
#         #standard + fgsm
#             for _dataset in datasets: 
#                 for _stop_val in stop_vals:
#                     for _ratio in ratio:
#                         for _ratio_adv in auxratioAdv:
#                             if _dataset == 'cifar10':
#                                 bounds = [4, 8, 12, 16]
#                             elif _dataset == 'mnist':
#                                 bounds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#                             elif _dataset == 'imageNet':
#                                 bounds = [2, 4, 8]
#                                 bounds = [2]
#                             elif _dataset == 'svhn':
#                                 bounds = [4, 8, 12, 16]
#                             else:
#                                 print("wrong dataset")
#                                 continue

#                             for _epsilon in bounds:
#                                 for lr in learning_rates:
#                                     for mm in momentums:
#                                         for bs in batchs:
#                                             for lr_adv in learning_rates_advs:
#                                                 for mm_adv in momentums_advs:
#                                                     for bs_adv in batchs_advs:
#                                                         command = "python3 train.py --type=both --alg=fgsm "

#                                                         command += " --dataset=" + _dataset 
#                                                         command += " --stop=" + str(stopCondition) + " --stop_val=" + str(_stop_val) 
#                                                         command += " --ratio=" + str(_ratio) 
#                                                         command += " --ratio_adv=" + str(_ratio_adv) 

#                                                         command += " --epsilon=" + str(_epsilon) 

#                                                         command += " --lr=" + str(lr) + " --momentum="  + str(mm) + " --batch=" + str(bs)
#                                                         command += " --lr_adv=" + str(lr_adv) + " --momentum_adv="  + str(mm_adv) + " --batch_adv=" + str(bs_adv)

#                                                         submit_job(command, available_workers, executor)


#         else:
#             print("wronh type of experiment")
                                    
# if __name__ == '__main__':
#     main()

# #parser.add_argument('--type', type=str, help='type of traing', default="std", choices=['std', 'robust', 'both'])
# #parser.add_argument('--alg', type=str, help='path to store the model', default="pgd", choices=['pgd', 'fgsm', 'pgd_rs', 'fgsm_rs'])

# #parser.add_argument('--ratio', type=float, help='percentage of data to train adversarial', default=1.0)
# #parser.add_argument('--epsilon', type=float, help='epsilon bound', default=0.1)

# #parser.add_argument('--num_iter', type=int, help='number of iterations for pgd ', default=10)
# #parser.add_argument('--alpha', type=float, help='alpha', default=0.01)

# #parser.add_argument('--dataset', type=str, help='dataset', default="mnist", choices=['mnist', 'cifar10', 'imageNet'])
# #parser.add_argument('--epochs', type=int, help='number of epochs', default=10)



# '''
# dataset = "svhn"
# import os
# arr = os.listdir()
# for i in arr:
#     if "model_" in i:
#         aux = "model" + dataset + i[5:]
#         os.rename(i, aux)



# import os
# arr = os.listdir()
# for i in arr:
#     if "ratioadv" not in i:
#         aux = i.split("_")
#         new_name = ""
#         for d in aux:
#             if "both" in i:
#                 if "ratio" not in d:
#                     new_name += d + "_"
#                 else:
#                     new_name += "ratio1.0"
#                     new_name += "_ratioadv" + str(d[5:]) + "_"
#             else:
#                 if "ratio" not in d:
#                     new_name += d + "_"
#                 else:
#                     new_name += "_ratioadv" + str(d[5:]) + "_"
#         os.rename(i, new_name[:-1])




# aux =

# '''

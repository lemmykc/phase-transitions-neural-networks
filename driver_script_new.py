from itertools import product
import os
import pandas as pd
import torch.multiprocessing
from torch.multiprocessing import Process, Manager, Pool
import classes
import numpy as np
import data_collector
#torch.multiprocessing.set_start_method('spawn')
import new_drivers
import HMC_inference
import model_setup
import time
import plotting_generators
import RLCT_estimate
import tall_samples

if __name__ == "__main__":
    
    ## SETUP META EXPERIMENT PARAMETERS
    
    # os.system("%logstart -o")
    
    eps_list = [0] 
    #symm_list = np.linspace(np.pi/4,np.pi/2, 30)
    #symm_list = [1.1, 1.2, 1.3]
    
    #symm_first = [0.7, 0.8, 0.9] #COME BACK TO THIS
    
    # EXP40 stuff 
    # symm_lin = np.linspace(1,1.36, 19)
    # symm_last = [1.43, 1.50, np.pi/2]
    # symm_list = np.concatenate([symm_lin, symm_last])
    
    symm_list=[0]
    
    #x_max_list = np.linspace(1,2,6)
    x_max_list = [1]
    #x_max_list = [1]
    #beta_list = [0.05, 0.1, 0.15]
    #beta_list = [0.1]
    
    num_hidden = [3]
    num_samples = [4000]
    #num_samples = [10000]
    #num_warmup = int(num_samples[0]*0.05)
    num_warmup = 1000
    #num_warmup = 100
    
    
    n = [10000]
    #beta_list = 1/(np.log(n[0])) * np.array([1/3, 2/3, 1, 4/3, 5/3])
    num_betas = 1
    prior_sd = [1]
    
    #q_scale_list = np.linspace(1,2,11)
    q_scale_list=[2.25, 2.5, 2.75, 3]
    #q_scale_list = [1]
    num_trials=4
    
    outlier_threshold = 5
    
    
    ## TEST FOR SINGLE EXPERIMENT
    # eps_list = [0]
    # symm_list = [0]
    # x_max_list = [1,1.2]
    # num_hidden = [3]
    # num_samples = [100]
    # n = [10000]
    # num_betas = 1
    # prior_sd=[1]
    # q_scale_list=[5]
    # num_trials = 3
    
    
    
    
    if num_betas>1:
        beta_list = np.squeeze(np.linspace(1 / np.log(n) * (1 - 1 / np.sqrt(2 * np.log(n))),
                                           1 / np.log(n) * (1 + 1 / np.sqrt(2 * np.log(n))), num_betas ))
    else:
        beta_list = [1/np.log(n[0])]
        
    silu_true = [False]

    
    # It is PIVOTAL to relabel this for each experiment - if it is not relabelled, then first_run=True can kill pre-existing data 
    experiment_number = 17
    
    # In order to run the experiments progressively I have split them up
    # en = experiment_number % 40
    # symm_list = symm_list[en*5:(en+1)*5]
    
    first_run = True
    
    
    
    directory = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments"

    parameters = {'eps_list': eps_list, 
                  'symm_list': symm_list, 
                  'x_max_list': x_max_list, 
                  'beta_list': beta_list, 
                  'num_hidden': num_hidden, 
                  'num_samples': num_samples,
                  'n': n,
                  'silu_true': silu_true,
                  'prior_sd': prior_sd,
                  'q_scale_list': q_scale_list,
                  }

    meta_experiment_data = {'experiment_number': experiment_number,
                            'num_trials': num_trials,
                            'directory': directory,
                            }
    
    # Create folders for data management
    if first_run:
        directory_dict, parameter_combinations = new_drivers.file_manager(parameters, meta_experiment_data,)
    else: 
        directory_dict = new_drivers.directory_dict_fn(meta_experiment_data)
        parameter_combinations = pd.read_csv(directory_dict['meta_data_path'])
        
        parameter_combinations = parameter_combinations.loc[parameter_combinations['Ln_w']==0]
    
        
    begin = time.perf_counter()
    
    # Setup multiprocessing
    manager = Manager()
    samples = manager.dict()
    pool = Pool(processes=4)
    jobs = []
    for row_index, pm in parameter_combinations.iterrows():
        # Initialise arguments for experiment
        args_trial = classes.Args(pm, directory_dict, num_warmup)
        
        # Multiprocessing exports CSV to relevant folder
        pool.apply_async(args_trial.model_driver)
        
    pool.close()
    pool.join() 
    end = time.perf_counter()
    print(f"Total time {end - begin:0.4f} seconds")
    
    # Run post-processing for statistical validation, produce initial density plots, etc.
    lims=([-3,2], [-2, 2])
    new_drivers.post_production_driver(experiment_number, num_trials, outlier_threshold=outlier_threshold,
                                       lims=lims, only_plot_betastar=False)
    
    
    
    ## Old post production stuff before driver
    # new_drivers.statistical_validation(directory_dict, num_trials)
    
    # tall_samples.pivot_driver(directory_dict['meta_experiment_folder'], only_bin=False)    
    
    # RLCT_estimate.rlct_estimates(directory_dict['meta_data_path'])
    
    # plot_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/Plots"
    
    # plot_dir= new_drivers.directory_creator(plot_path, "EXP{:0>2}".format(meta_experiment_data['experiment_number']))
    
    # for root, dirs, files in os.walk(directory_dict['validated_samples']):
    #     for f in files:
    #         beta_name = f[:-4]
    #         if beta_name[-2:] == "02":
    #             #lims =[(-1.5, 1.5), (-0.5,2)]
    #             plotting_generators.single_density(directory_dict['meta_experiment_folder'], beta_name, plot_dir, )
    #             plotting_generators.normal_unnormal(directory_dict['meta_experiment_folder'], beta_name, plot_dir, )
    
    
    
    
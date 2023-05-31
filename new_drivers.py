from itertools import product
import os
import pandas as pd
import torch.multiprocessing
from torch.multiprocessing import Process, Manager, Pool
import classes
import numpy as np
import data_collector
import tall_samples
import RLCT_estimate
import plotting_generators
#torch.multiprocessing.set_start_method('spawn')

def directory_creator(directory, new_subdir):
    # Creates a new directory if it doesn't already exist
    
    new_directory = directory + "/" + new_subdir
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        
    return new_directory

def radius_label(row, epsilon=0.5):
    ### ONLY FOR NUM_HIDDEN=2
    # Take in a validated_sample row and output the radius_label
    # epsilon is the radius threshold 
    
    r_1 = np.sqrt(row['wn_11']**2 + row['wn_12']**2)
    r_2 = np.sqrt(row['wn_21']**2 + row['wn_22']**2)
    
    rmin = min(r_1, r_2)
    rmax = max(r_1, r_2)
    
    # get label
    if rmin < epsilon:
        if 2 - epsilon < rmax < 2 + epsilon:
            label = 'degen'
        else: 
            label = 'outlier'
    elif 1 - epsilon < rmin < 1 + epsilon:
        if 1 - epsilon < rmax < 1 + epsilon:
            label = 'non_degen'
        else:
            label= 'outlier'
    else:
        label = 'outlier'
        
    return label
    

def directory_dict_fn(meta_experiment_data):
    # Create directory dictionary based on meta_exp_data dictionary that gets used in dirver
    
    directory_dict = {}
    directory_dict['meta_experiment_folder'] = directory_creator(meta_experiment_data['directory'], "EXP{:0>2}".format(meta_experiment_data['experiment_number']))
    directory_dict['raw_samples'] = directory_creator(directory_dict['meta_experiment_folder'], "raw_samples")
    directory_dict['validated_samples'] = directory_creator(directory_dict['meta_experiment_folder'], "validated_samples")
    directory_dict['true_parameters'] = directory_creator(directory_dict['meta_experiment_folder'], "true_parameters")
    
    meta_data_path = directory_dict['meta_experiment_folder'] + "/EXP{:0>2}_meta_data.csv".format(meta_experiment_data['experiment_number'])
    directory_dict['meta_data_path'] = meta_data_path
    directory_dict['diagnostics_txt_path'] = directory_dict['meta_experiment_folder'] + '/EXP{:0>2}_diagnostics.txt'.format(meta_experiment_data['experiment_number'])
    
    with open(directory_dict['diagnostics_txt_path'], 'w') as fp:
        pass
    
    return directory_dict

def file_manager(parameters, meta_experiment_data,):
    # Starting driver for creating meta_data file and parameter combinations
    
    directory_dict = directory_dict_fn(meta_experiment_data)
    
    # Create parameter combinations to iterate through 
    ## NOTE THE TRIALS MUST BE THE LAST ENTRY
    parameter_combinations = pd.DataFrame(list(product(parameters['eps_list'], 
                                                       parameters['symm_list'],
                                                       parameters['x_max_list'],
                                                       parameters['beta_list'],
                                                       parameters['num_hidden'],
                                                       parameters['num_samples'],
                                                       parameters['n'],
                                                       parameters['silu_true'],
                                                       parameters['prior_sd'],
                                                       parameters['q_scale_list'],
                                                       range(meta_experiment_data['num_trials']),
                                                       ))
                                          ,
                                          columns = ['eps_angle','symm_angle',
                                                     'x_max', 'beta', 'num_hidden', 
                                                     'num_samples', 'n', 'silu_true',
                                                     'prior_sd', 'q_scale',
                                                     'trial_num',])
    
    # Create experiment codes for metadata
    num_P = len(parameters['eps_list']) * len(parameters['symm_list']) * len(parameters['x_max_list']) * len(parameters['q_scale_list'])
    parameter_num_codes = pd.DataFrame(list(product([meta_experiment_data['experiment_number']],
                                                    range(num_P),
                                                    range(len(parameters['beta_list'])),
                                                    range(meta_experiment_data['num_trials']),
                                                    )), 
                                          columns = ['EXP', 'P', 'B','T'])
    parameter_combinations['exp_code'] = parameter_num_codes.apply(lambda row: 
                                                                      'EXP{:0>2}P{:0>3}B{:0>2}T{:0>2}'.format(
                                                                          row['EXP'], row['P'], row['B'], row['T']
                                                                          ), axis=1)
    # Create metadata file in experiment folder 
    experiments_parameter_CSV = parameter_combinations.copy()
    experiments_parameter_CSV['Ln_w'] = 0
    experiments_parameter_CSV['Ln_w_std'] = 0
    experiments_parameter_CSV['Ln_w_z_score']=0
    experiments_parameter_CSV['beta_inverse'] = experiments_parameter_CSV.apply(lambda row: 1/row['beta'], axis=1)
    
    experiments_parameter_CSV.to_csv(directory_dict['meta_data_path'])
    
    
    return directory_dict, parameter_combinations

def statistical_validation(directory_dict, num_trials, outlier_threshold=1.5, num_hidden=2):

    # Get experiment_meta_data    
    emd_df = pd.read_csv(directory_dict['meta_data_path'], index_col=[0])
   
    
    num_rows = len(emd_df.index)
    num_betas = int(num_rows/num_trials)

    
    for i in range(num_betas):
        # Get subarray of single betas i.e. all trials for P000B01 etc. 
        trial_subarray = emd_df[num_trials*i:(num_trials*(i+1))]
        Ln_w_array = np.array(trial_subarray['Ln_w'])
        
        std = Ln_w_array.std() 

        if std>0:
            z_scores = (Ln_w_array - Ln_w_array.mean())/std

            # VERY confusingly, .loc includes both endpoints
            emd_df.loc[num_trials*i:(num_trials*(i+1)-1),'Ln_w_z_score'] = z_scores
            
            trial_names_to_keep = list(trial_subarray.iloc[np.where(np.abs(z_scores)<outlier_threshold)[0]]['exp_code']) # THIS TAKES BOTH POSITIVE AND NEGATIVE OUTLIERS
            
            
            # Combine validated samples
            combined_df = pd.concat([pd.read_csv(directory_dict['raw_samples'] + "/" + name + ".csv", index_col=[0]) for name in trial_names_to_keep])
            
            # Remove any samples that fall beneath y<-1 TO GET RID OF -q VALUES
            # ALSO put in radius label 
            if num_hidden==2:
                combined_df['y_1'] = combined_df.apply(lambda row: row['w_12']*row['q_1'] ,axis=1)
                combined_df['y_2'] = combined_df.apply(lambda row: row['w_22']*row['q_2'] ,axis=1)
                combined_df = combined_df.loc[(combined_df['y_1'] > -1) | (combined_df['y_2'] > -1)]
                
                combined_df['radius_label'] = combined_df.apply(lambda row: radius_label(row) ,axis=1)
                
            
            # Save combined file
            exp_beta_name_short = trial_names_to_keep[0][:-3] # removes T00.
            csv_filename = directory_dict['validated_samples'] + "/" + exp_beta_name_short + ".csv"
            combined_df.to_csv(csv_filename)  
       
    # Update experiment_meta_data with z_scores
    emd_df.to_csv(directory_dict['meta_data_path'])
            
            
        
def post_production_driver(experiment_number, num_trials, outlier_threshold=1.5, lims=([-2,2],[-2,2]), val_finished=False, only_plot_betastar=True, directory = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments"):
    # Run statistical validation, pivot validated samples to long format, 
    # estimate RLCT, product plots
    
    
    meta_experiment_data = {'experiment_number': experiment_number,
                            'num_trials': num_trials,
                            'directory': directory,
                            }

    directory_dict = directory_dict_fn(meta_experiment_data)
    outlier_threshold = 1.5
    
    if not val_finished:
        statistical_validation(directory_dict, num_trials, outlier_threshold=outlier_threshold)
        
        tall_samples.pivot_driver(directory_dict['meta_experiment_folder'], only_bin=False)    
        
        RLCT_estimate.rlct_estimates(directory_dict['meta_data_path'])

    
    
    plot_path = directory + "/Plots"
    
    #plot_dir= directory_creator(plot_path, "EXP{:0>2}".format(meta_experiment_data['experiment_number']))
    
    
    meta_df = pd.read_csv(directory_dict['meta_data_path'], index_col=[0])
    
    exp_names = meta_df['exp_code']
    betas = np.unique([int(x[-4:-3]) for x in exp_names ])
    beta_mid = betas[int((len(betas)-1)/2)]
    
    
    for root, dirs, files in os.walk(directory_dict['validated_samples']):
        for f in files:
            
            beta_name = f[:-4]
            if only_plot_betastar:
                if beta_name[-1] == str(beta_mid):
                    plotting_generators.single_density(directory_dict['meta_experiment_folder'], beta_name, plot_path, lims )
            else:
                plotting_generators.single_density(directory_dict['meta_experiment_folder'], beta_name, plot_path, lims )
                #plotting_generators.normal_unnormal(directory_dict['meta_experiment_folder'], beta_name, plot_dir, )



    
        
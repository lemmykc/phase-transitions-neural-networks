"""Classes """

import torch
import numpy as np
# import drivers
# import free_energy
import HMC_inference
import model_setup
import data_collector
#import plotting
from functools import partial
import json 
import pandas as pd


class Args():  
    def __init__(self, 
                 parameters,
                 directory_dict,
                 num_warmup=False,
                 silu_param=30):
    
        self.num_samples = parameters['num_samples']
        if not num_warmup:
            self.num_warmup = int(self.num_samples*0.05)
        else:
            self.num_warmup = num_warmup
        #self.num_warmup=0
        self.num_data = parameters['n']
        self.num_input_nodes = 2
        self.num_output_nodes = 1
        self.num_hidden = parameters['num_hidden']
        self.num_hidden_true = self.num_hidden # since m=d
        self.prior_sd = parameters['prior_sd']
        
        
        self.eps_angle = parameters['eps_angle']
        self.x_max = parameters['x_max']
        self.symm_angle = parameters['symm_angle']
        self.beta = parameters['beta']
        self.q_scale = parameters['q_scale']
        
        self.exp_trial_code = parameters['exp_code']
        self.raw_samples_path = directory_dict['raw_samples']
        self.directory_dict = directory_dict
        
        
        self.target_accept_prob = 0.9
        self.chain_temp = self.num_data
        self.jit = True
        self.silu_param = silu_param
        
        
        if( parameters['silu_true'] ):
            self.nonlin = partial(self.silu, self.silu_param)
        else:
            self.nonlin = torch.nn.functional.relu
            
        self.X = None 
        self.Y = None
        self.samples = None
        
    def model_driver(self):
        # Get dataset D_n
        self.X, self.Y, self.w_0, self.b_0, self.q_0, self.c_0 = model_setup.get_data_true(self)
        
        # Get true param tensors
        true_params = {'w_0': self.w_0, 'b_0': self.b_0, 'q_0': self.q_0, 'c_0': self.c_0}
        
        # Run HMC sampler
        self.samples = HMC_inference.run_inference(model_setup.model, self)
        
        # Convert samples into dataframe format
        self.df = data_collector.samples_to_df(self)

        # Save data
        csv_path = self.raw_samples_path + "/" + self.exp_trial_code + ".csv"
        self.df.to_csv(csv_path, index=True)
        
        Ln_w_average = self.df['Ln_w'].mean()
        
        # Write average Ln_w to CSV
        meta_data_CSV = pd.read_csv(self.directory_dict['meta_data_path'], index_col=[0])
        meta_data_CSV.loc[meta_data_CSV['exp_code'] == self.exp_trial_code, 'Ln_w'] = Ln_w_average
        meta_data_CSV.to_csv(self.directory_dict['meta_data_path'])
        
        # Save true parameters for plotting
        true_param_path = self.directory_dict['true_parameters'] + "/" + self.exp_trial_code + "_w0.csv"
        self.true_params_df = data_collector.samples_to_df(self, w0_true = True)
        self.true_params_df.to_csv(true_param_path)

        
    @staticmethod
    def silu(beta, x):
        return x * torch.sigmoid( beta * x )

        
        

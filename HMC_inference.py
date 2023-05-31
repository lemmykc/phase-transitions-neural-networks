"""Perform HMC inference"""

import torch
import time
import numpy as np
import torch.multiprocessing
import sys
import contextlib, io
import pyro
import pyro.distributions as dist
from pyro.infer import HMC, MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_value, init_to_sample

#def generator(args):
    

# helper function for HMC inference
def run_inference(model, args):
    H = args.num_hidden
    X = args.X
    Y = args.Y
    
    beta = args.beta
    
    # MUST FIX
    # init_dict = {"w": args.w_0.clone().detach() , "b": args.b_0.clone().detach(),
    #              "q": args.q_0.clone().detach(), "c": args.c_0.clone().detach()}
        
    start = time.time()
    kernel = NUTS(model, adapt_step_size=True, 
                target_accept_prob = args.target_accept_prob,
                jit_compile=args.jit, 
                #init_strategy=init_to_value(values=init_dict)
                )
    mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.num_warmup)
    mcmc.run(X, Y, H, beta, args.prior_sd, args.nonlin)
    print("\n[beta = {}]".format(beta))
    print(args.exp_trial_code)
    mcmc.summary(prob=0.5)
    # print(mcmc.diagnostics())
    
    # f = io.StringIO()
    # with contextlib.redirect_stdout(f):
        
    
    # sys.stdout = open(args.directory_dict['diagnostics_txt_path'], 'a') 
    # print("---------------------------------------------------------------------------------")
    # print("\n[beta = {}]".format(beta))
    # print(args.exp_trial_code)
    # mcmc.summary(prob=0.5)
    # print("---------------------------------------------------------------------------------")
    # sys.stdout.close()
    
   
    return mcmc.get_samples()


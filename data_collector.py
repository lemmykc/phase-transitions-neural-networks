"""Data management of samples """

import torch
import numpy as np
import pandas as pd

def neg_log_loss(X, Y, samples, num_samples, nonlin, M, w0_true):
    # Calculates negative log loss L_n(w) for each sample in samples
    
    nll_list = []
    for i in range(num_samples):
        if w0_true: 
            w = samples['w']
            b = samples['b']
            q = samples['q']
            c = samples['c']
        else:
            w = samples['w'][i]
            b = samples['b'][i] 
            q = samples['q'][i] 
            c = samples['c'][i] 
        
        
        a = torch.matmul(X, w) + b
        f = torch.matmul(nonlin(a), q) + c

        MSE = np.square(np.subtract(Y,f)).mean()
        
        Ln_w = M/2 * np.log(2*np.pi) + 1/2 * MSE 
        Ln_w = Ln_w.item()

        nll_list.append(Ln_w)
        
    return nll_list

def samples_to_df(args, w0_true = False):
    ## Converts the samples from HMC into a usable dataframe to perform data analysis
    
    N, M = args.X.shape[1], args.Y.shape[1]
    
    X = args.X
    Y = args.Y
    
    if args.num_hidden == 2:
        col_names = ["w_11", "w_12", "w_21", "w_22", "b_1", 
                     "b_2", "q_1", "q_2", "c", "wn_11", "wn_12", 
                     "wn_21", "wn_22", "Ln_w", "beta", "eps_angle", "symm_angle",
                     "x_max", 'q_scale']
    elif args.num_hidden ==3:
        col_names = ["w_11", "w_12", "w_21", "w_22", "w_31", "w_32", "b_1", 
                     "b_2", "b_3", "q_1", "q_2", "q_3", "c", "wn_11", "wn_12", 
                     "wn_21", "wn_22", "wn_31", "wn_32", "Ln_w", "beta", "eps_angle", 
                     "symm_angle", "x_max",'q_scale']
    elif args.num_hidden ==4:
        col_names = ["w_11", "w_12", "w_21", "w_22", "w_31", "w_32", "w_41", 
                     "w_42", "b_1", "b_2", "b_3", "b_4", "q_1", "q_2", "q_3", 
                     "q_4", "c", "wn_11", "wn_12", "wn_21", "wn_22", "wn_31", 
                     "wn_32", "wn_41", "wn_42", "Ln_w", "beta", "eps_angle", "symm_angle",
                     "x_max",'q_scale']
    
    if w0_true: 
        samples = {'w': args.w_0, 'b': args.b_0, 'q': args.q_0, 'c':args.c_0}
        num_samples = 1

        
        
        w_np = torch.flatten(samples['w'].T).numpy()
        b_np = torch.flatten(samples['b']).numpy()
        q_np = torch.flatten(samples['q']).numpy()
        c_np = torch.flatten(samples['c']).numpy()

        
        w_0 = samples['w'].T.numpy()
        q_0 = samples['q'].numpy()

        w_normalise = np.multiply(q_0, w_0)
        w_normalise_flatten_np = w_normalise.flatten()

        nll_list = neg_log_loss(X, Y, samples, num_samples, args.nonlin, M, w0_true)
        
        full_array = np.hstack((w_np,b_np,q_np,c_np, w_normalise_flatten_np, 
                                nll_list[0], args.beta, args.eps_angle, 
                                args.symm_angle, args.x_max, args.q_scale))

        df = pd.DataFrame([full_array], columns = col_names)
        
    else:
        samples = args.samples
        num_samples = args.num_samples
        
        # Reshape w into (samples, [w_11, w_12, ..., w_42])
        w_flatten = torch.flatten(torch.transpose(samples['w'] ,1,2), 1,2)
        # b is already in this shape
        q_flatten = torch.flatten(samples['q'], 1, 2)
        # c is already in this shape 
        
        w_np = w_flatten.numpy()
        b_np = np.squeeze(samples['b'].numpy())
        q_np = q_flatten.numpy()
        c_np = samples['c'].numpy()
        
        
        # Calculate w_hat values (normalise by q) for plotting
        w_normalise = torch.einsum('ijk,ikm->ijk', samples['w'], torch.abs(samples['q']))
        w_normalise_flatten_np = torch.flatten(torch.transpose(w_normalise ,1,2), 1,2).numpy()
        
        nll_list = neg_log_loss(X, Y, samples, num_samples, args.nonlin, M, w0_true)

        
        full_array = np.concatenate([w_np, b_np, q_np, 
                                     c_np, w_normalise_flatten_np,
                                     np.array(nll_list)[:, None],
                                     args.beta*np.ones((num_samples, 1)),
                                     args.eps_angle*np.ones((num_samples, 1)),
                                     args.symm_angle*np.ones((num_samples, 1)),
                                     args.x_max*np.ones((num_samples, 1)),
                                     args.q_scale*np.ones((num_samples, 1)),
                                     ],
                                    axis=1)
        
        df = pd.DataFrame(full_array, columns = col_names)
    
    
    return df
        
    
    
    


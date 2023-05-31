"""Functions to set up true networks, and the model to sample from in HMC"""

import numpy as np
import torch
import torch.multiprocessing
import pyro
import pyro.distributions as dist
import sys



def model(X, Y, H, beta, prior_sd, nonlin):
    # Define the model and priors using pyro for HMC
    # H is number of model nodes (d in thesis)
    
    M, N = X.shape[1], Y.shape[1]

    # w is the weight matrix R^2 --> R^H
    w = pyro.sample("w", dist.Normal(torch.zeros((M, H)), prior_sd * torch.ones((M, H)))) 
    
    # b is the bias in the hidden layer
    b = pyro.sample("b", dist.Normal(torch.zeros(H), prior_sd * torch.ones(H)))
    
    # q is the weight matrix R^H --> R^1
    q = pyro.sample("q", dist.Normal(torch.zeros((H, N)), prior_sd * torch.ones((H, N))))
    
    #q = pyro.sample("q", dist.HalfNormal(prior_sd * torch.ones((H, N))))
    
    # c is the final bias
    c = pyro.sample("c", dist.Normal(torch.zeros(N), prior_sd * torch.ones(N)))
    
    a = torch.matmul(X, w) + b
    f = torch.matmul(nonlin(a), q) + c
     
    return pyro.sample("Y", dist.Normal(f, 1/np.sqrt(beta)), obs=Y)


def true_symmetric_params(num_hidden_true):
    ## Defines true parameters of an m-symmetric network
    
    # Output: true symmetric parameters with no perturbation as np.arrays
    
    # Construct symmetric w's, which are roots of unity. w_0 is in the first quadrant
    # and rotation proceeds clockwise from there (I think this is an artefact
    # of how the rotation matrices are applied)
    
    angle = 2 * np.pi / num_hidden_true
    w_init = np.array([[np.cos(angle/2), np.sin(angle/2)]]) #w_0 in paper 
    
    w_list = [ np.matmul(w_init, np.array([[np.cos(k*angle), -np.sin(k*angle)],
                                             [np.sin(k*angle), np.cos(k*angle)]])) for k in range(num_hidden_true)]
    
    w = np.vstack(w_list)
    w = np.transpose(w)
    
    
    # Other true parameters, invariant between experiments 
    b = np.array([-0.3 * np.ones((num_hidden_true))])
    q = np.ones((num_hidden_true,1))
    c = np.array([0.0])
    
    return w, b, q, c

def true_params_perturb_one(num_hidden_true, eps_angle):
    ## Rotate one node

    # Get true symmetric parameters with no perturbation
    w, b, q, c = true_symmetric_params(num_hidden_true)
    
    # Rotate first weight by eps_angle 
    w_altered_index = 0 # to keep consistency
    rotation_matrix = np.array([[np.cos(eps_angle), -np.sin(eps_angle)],
                                [np.sin(eps_angle), np.cos(eps_angle)]])
    w[:,w_altered_index] = rotation_matrix @ w[:,w_altered_index]
    
    return w, b, q, c

def true_params_perturb_two(num_hidden_true, symm_angle):
    # Rotate two nodes symmetrically
    # This is used in PT1 in my thesis - deforming to node degeneracy
    
    # Get true symmetric parameters with no perturbation
    w, b, q, c = true_symmetric_params(num_hidden_true)
    
    # Symmetrically perturb two node weights 
    rotation_matrix_pos = np.array([[np.cos(symm_angle), -np.sin(symm_angle)],
                                [np.sin(symm_angle), np.cos(symm_angle)]])
    
    rotation_matrix_neg = np.array([[np.cos(-symm_angle), -np.sin(-symm_angle)],
                                [np.sin(-symm_angle), np.cos(-symm_angle)]])
    
    
    if num_hidden_true==2:
        w = np.array([[1,-1],[0,0]], dtype="float")
        w[:, 0] = rotation_matrix_pos @ w[:, 0]
        w[:, 1] = rotation_matrix_neg @ w[:, 1]
    else:
        # EDITED JUST FOR NOW
        w[:, 2] = rotation_matrix_neg @ w[:,2]
        w[:, 3] = rotation_matrix_pos @ w[:,3]
    
    
    
    return w, b, q, c


def true_params_q_scale(num_hidden_true, q_scale):
    ## PT2 - remove orientation reversing symmetry by increasing q gradient
    
    w, b, q, c = true_symmetric_params(num_hidden_true)
    q[2,0]= q_scale
    
    return w,b,q,c


def get_data_true(args):
    # Output: Dataset D_n=(X,Y), true parameter tensors given num_hidden and 
    # perturbation parameters
    
    num_data = args.num_data
    M = args.num_input_nodes
    N = args.num_output_nodes
    
    
    # set up true parameter tensors
    # NOTE: These arguments are specific to the experiments in my thesis, 
    # i.e. whenever I ran a two node experiment, it was always to deform to degeneracy
    if args.num_hidden == 2:
        w, b, q, c = true_params_perturb_two(2, symm_angle = args.symm_angle)
        
    elif args.num_hidden == 3: 
        # w, b, q, c = true_params_perturb_one(3, eps_angle = args.eps_angle) # Old attempt to rotate one node to destroy orientation reversing symmetry
        
        # Better attempt to alter q gradient
        w, b, q, c = true_params_q_scale(3, args.q_scale)
        
    elif args.num_hidden == 4:
        w, b, q, c = true_params_perturb_two(4, symm_angle = args.symm_angle)
    else: 
        print("The code only supports d=2, 3 or 4 at this time")
        sys.exit()
        
    w_t = torch.tensor(w, dtype=torch.float)
    b_t = torch.tensor(b, dtype=torch.float)
    q_t = torch.tensor(q, dtype=torch.float)
    c_t = torch.tensor(c, dtype=torch.float)
    
    
    # Sample X uniformly from q(x) = [-x_max, x_max]^2
    X = 2 * args.x_max * torch.rand(num_data, M) - args.x_max 
    
    # sample from regression model p(y|x,w) with variance=1
    a = torch.matmul(X, w_t) + b_t
    f = torch.matmul(args.nonlin(a), q_t) + c_t 
    
    ydist = dist.Normal(f, 1)
    Y = ydist.sample()
    
    
    return X, Y, w_t, b_t, q_t, c_t



        
        
        
        
        
        



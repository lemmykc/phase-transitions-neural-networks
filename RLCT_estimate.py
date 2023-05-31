import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


def get_gradient(df):

    X = df['beta_inverse'].values
    Y = df['mean'].values # This is mean Ln_w
    gradient = LinearRegression().fit(X.reshape(-1,1), Y).coef_
    
    return gradient[0]
    

def rlct_estimates(exp_meta_data_path, outlier_threshold=1.5):
    # INPUT: exp_meta_data_path of csv with average Ln_w from each trial 
    
    
    df = pd.read_csv(exp_meta_data_path, index_col=[0])
    df['Ln_w'] = df['Ln_w']*10000
    df = df.loc[df['Ln_w']>0]
    
    # if 'Ln_w_z_score' not in df.columns:
    #     df = df.loc[df['Ln_w']>0]
    #     df = df.groupby(level = df.index.names - ['trial', 'Ln_w', 'num_trials', 'exp_code']).mean().reset_index()
    #     #df
    
    
    df = df.loc[np.abs(df['Ln_w_z_score'])<outlier_threshold] # get only validated samples
    
    
    
    df_grp = df.groupby(['symm_angle', 'beta_inverse'])
    df_agg = df_grp['Ln_w'].agg([np.mean, np.std]).reset_index()
    
    df_rlcts = df_agg.groupby('symm_angle').apply(get_gradient)
    
    meta_exp_path = os.path.dirname(exp_meta_data_path)
    exp_code = meta_exp_path[-5:]
    df_rlcts.to_csv(meta_exp_path + "/" + exp_code + "_rlct.csv")
    
    


# exp_meta_data_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP30/EXP30_meta_data.csv"
# rlct_estimates(exp_meta_data_path)



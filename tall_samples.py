import pandas as pd
import numpy as np
import new_drivers
import os

def angle(y,x):
    angle = np.arctan2(y,x)
    if -np.pi < angle < -np.pi/2:
        angle += 2*np.pi
    return angle

def angle_eps(angle):
    
    if -np.pi/6 <= angle < np.pi/6 or np.pi/2 <= angle < 5*np.pi/6 or 7*np.pi/6 <= angle < 3*np.pi/2:
        label = -1
    else: 
        label = 1
    
    return label

def new_V(y, x, angle):
    r = np.sqrt(y**2+x**2)
    V = 1/(1+np.exp(-13*(r-0.5))) * (angle + np.pi+2)
    
    return V

def pivot_driver(exp_path, only_bin=False, num_hidden=False):
    
    
    experiment_name = exp_path[-5:]
    exp_meta_data = pd.read_csv(exp_path + "/" + experiment_name + "_meta_data.csv", index_col=[0])
    
    validated_samples_path = exp_path + "/validated_samples"
    long_samples_path = new_drivers.directory_creator(exp_path, "long_samples")
    
    long_samples_V_angle_path = new_drivers.directory_creator(exp_path, "long_binned_V_angle")
    long_samples_V_new_path = new_drivers.directory_creator(exp_path, "long_binned_V_new")
    
    # walk through samples 
    for root, dirs, files in os.walk(validated_samples_path):
        for sample_csv in files: 
            if sample_csv[-4:] == ".csv":
                exp_code = sample_csv[:-4]
                sample_csv_path = os.path.join(root, sample_csv)
                samples = pd.read_csv(sample_csv_path, index_col = [0])
                
                
                
                if not only_bin:
                    df = pivot_long(samples, exp_meta_data, exp_code, num_hidden=num_hidden)
                    df.to_csv(long_samples_path + "/" + exp_code + "_long.csv")
                else: 
                    df = pd.read_csv(long_samples_path + "/" + exp_code + "_long.csv", index_col=[0])
                
                
                df_long_V_angle = bin_free_energy(df, exp_meta_data, False)
                df_long_V_angle.to_csv(long_samples_V_angle_path + "/" + exp_code + "_long_V_angle.csv")
                
                df_long_V_new = bin_free_energy(df, exp_meta_data, True)
                df_long_V_new.to_csv(long_samples_V_new_path + "/" + exp_code + "_long_V_new.csv")
                    
                
    

def pivot_long(samples, exp_meta_data, beta_code, num_hidden=False, normalise=True, true_nodes = False):

    if not num_hidden:
        exp_code = beta_code + "T00"
        num_hidden = int(exp_meta_data.loc[exp_meta_data['exp_code']==exp_code, 'num_hidden'])

    
    if normalise:
        x_cols = ['wn_{}1'.format(n+1) for n in range(num_hidden)]
        y_cols = ['wn_{}2'.format(n+1) for n in range(num_hidden)]
    else:
        x_cols = ['w_{}1'.format(n+1) for n in range(num_hidden)]
        y_cols = ['w_{}2'.format(n+1) for n in range(num_hidden)]
        
    q_cols = ['q_{}'.format(n+1) for n in range(num_hidden)]
    b_cols = ['b_{}'.format(n+1) for n in range(num_hidden)]
    
    x_list = []
    y_list = []
    q_list = []

    eps_sign_list = []
    

    
    for i, row in samples.iterrows():
        x_list.extend(list(row[x_cols]))
        y_list.extend(list(row[y_cols]))
        q_list.extend(list(row[q_cols]))
        #q_sign_list.extend(list(np.sign(row[q_cols])))
        eps_sign_list.extend(list(-np.sign(row[b_cols])))

    index_list = np.tile(np.arange(1, num_hidden+1), len(samples.index))
    Ln_w_list = np.repeat(samples['Ln_w'], num_hidden)
    
    
    
    # df = pd.DataFrame({'w_i1': x_list, 'w_i2': y_list, 'q_i': q_list, 
    #                    #'q_sign': q_sign_list, 
    #                    'eps_sign': eps_sign_list, 'index': index_list, 
    #                    'Ln_w': Ln_w_list, 'radius_label':radius_label_list})
    
    df = pd.DataFrame({'w_i1': x_list, 'w_i2': y_list, 'q_i': q_list, 
                       #'q_sign': q_sign_list, 
                       'eps_sign': eps_sign_list, 'index': index_list, 
                       'Ln_w': Ln_w_list})
    

    for col in samples.columns:
        if col[:12]=='radius_label':
            if true_nodes:
                df[col] = 'None'
            elif num_hidden==2:
                df[col] = np.repeat(samples[col], num_hidden)
            else: 
                df[col] = 'None'
    

    df['q_sign'] = df.apply(lambda row: np.sign(row['q_i']), axis=1)
    df['V_angle'] = df.apply(lambda row: angle(row['w_i2'], row['w_i1']), axis=1)
    df['V_new'] = df.apply(lambda row: new_V(row['w_i2'], row['w_i1'], row['V_angle']), axis=1)
    df['index_eps'] = df.apply(lambda row: int(row['eps_sign']* row['index']), axis=1)
    
    
    df['x_max'] = float(samples['x_max'].iloc[0])
    df['eps_angle'] = float(samples['eps_angle'].iloc[0])
    df['symm_angle'] = float(samples['symm_angle'].iloc[0])
    
    if 'q_scale' in samples.columns:
        df['q_scale'] = np.repeat(list(samples['q_scale']), num_hidden)  
                              
    #float(samples['q_scale'].iloc[0])
    
    df['angle_label'] = df.apply(lambda row: angle_eps(row['V_angle']), axis=1)
    
    
    return df



def bin_free_energy(df, exp_meta_data, new_V_true):
    
    if new_V_true: 
        bins = np.linspace(0, 10, 100)
        V_mean_df = df.groupby(np.digitize(df['V_new'], bins)).mean()[['V_new', 'Ln_w']]
        V_mean_df.rename({'Ln_w': 'Ln_w_mean'})
        
        V_std_col = df.groupby(np.digitize(df['V_new'], bins)).std()[['Ln_w']]
        V_mean_df['Ln_w_std'] = V_std_col
        
    else:
        bins = np.linspace(-np.pi/2, 3*np.pi/2, 100)
        V_mean_df = df.groupby(np.digitize(df['V_angle'], bins)).mean()[['V_angle', 'Ln_w']]
        V_mean_df.rename(columns = {'Ln_w': 'Ln_w_mean'}, inplace=True)
        
        V_std_col = df.groupby(np.digitize(df['V_angle'], bins)).std()[['Ln_w']]
        V_mean_df['Ln_w_std'] = V_std_col

        
    return V_mean_df
                                
        
                      
# exp_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP09"

# pivot_driver(exp_path, only_bin=False)                     
                      
        
        
        
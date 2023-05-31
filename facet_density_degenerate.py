import pandas as pd
import numpy as np
import new_drivers
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
import plotting
import tall_samples
import os

def facet_densities(exp_dir_list):
    
    orientation = 'portrait'
    
    df_list = [pd.read_csv(exp_dir, index_col=[0]) for exp_dir in exp_dir_list]
    df = pd.concat(df_list)
    
    true_samples_list = []
    
    for ed in exp_dir_list:
        
        exp_dir = ed[:98]
        beta_name = ed[-21:-9]
        exp_meta_data = pd.read_csv(exp_dir + "/" + beta_name[:5] + "_meta_data.csv", index_col=[0])
        # Get true parameters
        true_samples = pd.read_csv(exp_dir + "/true_parameters/" + beta_name + "T00_w0.csv")
        true_samples['q_scale'] = 0
        true_samples_long = tall_samples.pivot_long(true_samples, exp_meta_data, beta_name, normalise=True, true_nodes=True)
        true_samples_list.append(true_samples_long)
    
    df_true = pd.concat(true_samples_list)
    
    df = df.round({'symm_angle': 2})
    df_true = df_true.round({'symm_angle':2})
    
    df = df.rename(columns = {"symm_angle":'theta'})
    df_true = df_true.rename(columns = {"symm_angle":'theta'})
    
    
    
    # df = df.rename(columns = {"symm_angle":'$\theta$', 'w_i1': '$w_{i,1}$', 'w_i2': '$w_{i,2}$'})
    # df_true = df_true.rename(columns = {"symm_angle":'$\theta$', 'w_i1': '$w_{i,1}$', 'w_i2': '$w_{i,2}$'})
    
    #print(df.columns)
    #print(df_true.columns)
    
    #print(df_true)
    
    
    #df = df.sample(frac=0.01, replace=False)
    
    # print(df)
    # print(df['symm_angle'].value_counts())
    
    if orientation == 'landscape':
        plt.rc('figure', figsize=(11.69,8.27))
        g = sns.displot(df, x = "w_i1", y = "w_i2", clip=([-0.75, 0.75], [-0.7, 2.7]), col='theta', col_wrap=4, kind='kde')
        
    
    if orientation == 'portrait':
        plt.rc('figure', figsize=(8.27,11.69))
        g = sns.displot(df, x = "w_i1", y = "w_i2", clip=([-0.75, 0.75], [-0.7, 2.7]), col='theta', col_wrap=2, kind='kde')
        
    
    #g = sns.displot(df, x = "$w_{i,1}$", y = "$w_{i,2}$", clip=([-0.75, 0.75], [-0.7, 2.7]), col='$\theta$', col_wrap=4, kind='kde')
    
    
    #g = sns.displot(df, x = "w_i1", y = "w_i2", clip=([-0.75, 0.75], [-0.7, 2.7]), col='theta', col_wrap=4, kind='kde')
    
    
    # ax = plt.gca()
    # ax.plot(df['w_i1'], df['w_i2'], 'o', color='red')
    #g.map_dataframe(sns.kdeplot, x = "w_i1", y = "w_i2", clip=([-0.75, -0.75], [-0.7, 2.7]))
    
    # g.axes[0,0].set_ylabel(r'$w_{i,2}$',fontsize=18)
    # g.axes[1,0].set_ylabel(r'$w_{i,2}$',fontsize=18)
    
    axes = g.axes.flatten()
    
    for ax in axes:
        
        symm_angle = ax.get_title().split(' = ')[1]
        #print(symm_angle)
        ax.set_title(r'$\theta$={}'.format(symm_angle))
        df_true_small = df_true.loc[df_true['theta']==float(symm_angle)]
        #print(df_true_small)
        ax.plot(df_true_small['w_i1'], df_true_small['w_i2'], 'o', color='red', markersize=13)
        
        # ax.set_xlabel(r'$w_{i,1}$',fontsize=18)
        
    
        
    g.set_axis_labels(x_var=r'$\hat{w}_{i,1}$', y_var=r'$\hat{w}_{i,2}$', fontsize=18)
    #g.axes[0,0]df.set_xlabel('axes label 1')
    
    #g.tight_layout() 
    g.savefig("density_facet_GOOD_landscape_all_samples.png")
    
    # fig = plt.get_figure()
    # fig.savefig("density_facet_test.png")
        
        
if True:
    exp_dir_list = ['/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP20/long_samples/EXP20P004B02_long.csv',
                '/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP21/long_samples/EXP21P003B02_long.csv',
                "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP22/long_samples/EXP22P000B02_long.csv",
                "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP22/long_samples/EXP22P002B02_long.csv",
                "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP22/long_samples/EXP22P004B02_long.csv",
                "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP23/long_samples/EXP23P001B02_long.csv",
                "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP24/long_samples/EXP24P000B00_long.csv",
                "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP24/long_samples/EXP24P002B00_long.csv"]
        
    facet_densities(exp_dir_list)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
import tall_samples
#%%

# meta_exp_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP51"

# tall_samples.pivot_driver(meta_exp_path, only_bin=False, num_hidden=3)


#%%


#%%   

def facet_densities_eps(exp_dir_list):
    
    orientation = 'portrait'
    
    df_list = [pd.read_csv(exp_dir, index_col=[0]) for exp_dir in exp_dir_list]
    df = pd.concat(df_list)
    
    true_samples_list = []
    
    for ed in exp_dir_list:
        
        exp_dir = ed[:97]
        beta_name = ed[-21:-9] # THIS DEPENDNED ON THE EXP FOR SOME REASON
        
        #exp_meta_data = pd.read_csv(exp_dir + "/" + beta_name[:5] + "_meta_data.csv", index_col=[0])
        # Get true parameters
        
        true_samples = pd.read_csv(exp_dir + "/true_parameters/" + beta_name + "T00_w0.csv")
        true_samples_long = tall_samples.pivot_long(true_samples, [], [], num_hidden=3, normalise=True, true_nodes=True)
        true_samples_list.append(true_samples_long)
    
    df_true = pd.concat(true_samples_list)
    
    # df = df.round({'symm_angle': 2})
    # df_true = df_true.round({'symm_angle':2})
    
    # df = df.rename(columns = {"symm_angle":'theta'})
    # df_true = df_true.rename(columns = {"symm_angle":'theta'})
    
    
    
    # df = df.rename(columns = {"symm_angle":'$\theta$', 'w_i1': '$w_{i,1}$', 'w_i2': '$w_{i,2}$'})
    # df_true = df_true.rename(columns = {"symm_angle":'$\theta$', 'w_i1': '$w_{i,1}$', 'w_i2': '$w_{i,2}$'})
    
    #print(df.columns)
    #print(df_true.columns)
    
    #print(df_true)
    
    
    #df = df.sample(frac=0.01, replace=False)
    
    # print(df)
    # print(df['symm_angle'].value_counts())
    clip = ([-3, 3], [-2, 2])
    
    if orientation == 'landscape':
        plt.rc('figure', figsize=(11.69,8.27))
        g = sns.displot(df, x = "w_i1", y = "w_i2", clip=clip, 
                        col='q_scale', col_wrap=3, kind='kde',
                        hue='eps_sign',
                        common_norm=True,
                        palette='viridis')
        
    
    if orientation == 'portrait':
        plt.rc('figure', figsize=(8.27,11.69))
        g = sns.displot(df, x = "w_i1", y = "w_i2", clip=clip, col='x_max', 
                        col_wrap=2, kind='kde')
        
    
    #g = sns.displot(df, x = "$w_{i,1}$", y = "$w_{i,2}$", clip=([-0.75, 0.75], [-0.7, 2.7]), col='$\theta$', col_wrap=4, kind='kde')
    
    
    #g = sns.displot(df, x = "w_i1", y = "w_i2", clip=([-0.75, 0.75], [-0.7, 2.7]), col='theta', col_wrap=4, kind='kde')
    
    
    # ax = plt.gca()
    # ax.plot(df['w_i1'], df['w_i2'], 'o', color='red')
    #g.map_dataframe(sns.kdeplot, x = "w_i1", y = "w_i2", clip=([-0.75, -0.75], [-0.7, 2.7]))
    
    # g.axes[0,0].set_ylabel(r'$w_{i,2}$',fontsize=18)
    # g.axes[1,0].set_ylabel(r'$w_{i,2}$',fontsize=18)
    
    axes = g.axes.flatten()
    
    for ax in axes:
        
        x_max = ax.get_title().split(' = ')[1]
        #print(symm_angle)
        ax.set_title(r'$a$={}'.format(x_max))
        df_true_small = df_true.loc[df_true['x_max']==float(x_max)]
        #print(df_true_small)
        ax.plot(df_true_small['w_i1'], df_true_small['w_i2'], 'o', color='red', markersize=13)
        ax.set_xlim([-2,2])
        ax.set_ylim([-2, 2])
        # ax.set_xlabel(r'$w_{i,1}$',fontsize=18)
        
    
        
    g.set_axis_labels(x_var=r'$\hat{w}_{i,1}$', y_var=r'$\hat{w}_{i,2}$', fontsize=18)
    #g.axes[0,0]df.set_xlabel('axes label 1')
    
    #g.tight_layout() 
    g.savefig("density_xmax_PT_facet_portrait_Exp26.png")
    
    
#%%

# long_samps_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP51/long_samples/EXP51P{:0>3}B00T00_long.csv"
# exp_dir_list = [long_samps_path.format(6*i) for i in range(6)]

long_samps_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP51/long_samples/EXP51P{:0>3}B00T00_long.csv"
exp_dir_list = [long_samps_path.format(i) for i in range(6)]



facet_densities_eps(exp_dir_list)

    
#%%

exp15_samples_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP15/raw_samples"
trial_dict = {0: [0,1,2,3], 
              3: [0,1,2],
              5: [0,1,2],
              7: [0,1,2,3],
              10: [0,1,2]}
exp_codes = []
for k in [0, 3, 5, 7, 10]:
    exp_codes.extend(["EXP15P0{:0>2}B00T{:0>2}".format(str(k), str(i)) for i in trial_dict[k]])

exp_dfs = [pd.read_csv(exp15_samples_path + "/" + code + ".csv", index_col=[0]) for code in exp_codes]



exp16_samples_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP16/raw_samples"
exp_codes_16 = ["EXP16P000B00T{:0>2}".format(i) for i in [0,1,2,3]]
exp_dfs.extend([pd.read_csv(exp16_samples_path + "/" + code + ".csv", index_col=[0]) for code in exp_codes_16])


raw_dfs = pd.concat(exp_dfs)

df_long = tall_samples.pivot_long(raw_dfs, [], [], num_hidden=3, normalise=True, true_nodes = False)
df_long.to_csv("long_samples_Exp15_16_epsilon_facet.csv", index=True)




#%%

# EXP 15 + 16 FACET DENSITIES FINAL 

exp15_true_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP15/true_parameters"
exp16_true_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP16/true_parameters"

true_sample_15_codes = ["EXP15P0{:0>2}B00T00".format(str(i)) for i in [0, 3, 5, 7, 10]]
true_dfs = [pd.read_csv(exp15_true_path + "/" + code + "_w0.csv", index_col=[0]) for code in true_sample_15_codes]

true_sample_16_code = "EXP16P000B00T00"
true_dfs.append(pd.read_csv(exp16_true_path + "/" + true_sample_16_code + "_w0.csv", index_col=[0]))

true_dfs_concat = pd.concat(true_dfs)

df_long_true = tall_samples.pivot_long(true_dfs_concat, [], [], num_hidden=3, normalise=True, true_nodes = True)




#%%

# EPSILON PLOT FOR 15 + 16

df = pd.read_csv("long_samples_Exp15_16_epsilon_facet.csv", index_col=[0])
df_true = df_long_true

#df = df.sample(frac=0.05)


df = df.round({'q_scale': 2})
df_true = df_true.round({'q_scale':2})

orientation = 'portrait'
clip = ([-5, 5], [-5, 5])

if orientation == 'landscape':
    plt.rc('figure', figsize=(11.69,8.27))
    g = sns.displot(df, x = "w_i1", y = "w_i2", clip=clip, 
                    col='q_scale', col_wrap=3, kind='kde',
                    hue='eps_sign',
                    common_norm=True,
                    palette='viridis')
    

if orientation == 'portrait':
    plt.rc('figure', figsize=(8.27,11.69))
    g = sns.displot(df, x = "w_i1", y = "w_i2", clip=clip, col='q_scale', 
                    col_wrap=2, kind='kde')
    
    
axes = g.axes.flatten()

for ax in axes:
    
    q_scale = ax.get_title().split(' = ')[1]
    #print(symm_angle)
    ax.set_title(r'$\vartheta$={}'.format(q_scale))
    df_true_small = df_true.loc[df_true['q_scale']==float(q_scale)]
    #print(df_true_small)
    ax.plot(df_true_small['w_i1'], df_true_small['w_i2'], 'o', color='red', markersize=13)
    ax.set_xlim([-3,2])
    ax.set_ylim([-2, 2])
    # ax.set_xlabel(r'$w_{i,1}$',fontsize=18)
    

    
g.set_axis_labels(x_var=r'$\hat{w}_{i,1}$', y_var=r'$\hat{w}_{i,2}$', fontsize=18)
#g.axes[0,0]df.set_xlabel('axes label 1')

#g.tight_layout() 
g.savefig("density_epsilon_PT_facet_portrait_Exp15_16.png")






#%% 

# X MAX PHASE TRANSITION

long_samps_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP26/long_samples/EXP26P{:0>3}B00_long.csv"
exp_dir_list = [long_samps_path.format(i) for i in range(6)]



facet_densities_eps(exp_dir_list)





    
    
#%%

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

#%%
import new_drivers
import tall_samples

#%%

def radius_PT_df_generator(exp_num, is_24=False):
    exp_num_str = "EXP{:0>2}".format(exp_num)
    exp_dir = '/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/' + exp_num_str
    long_samps_dir = exp_dir + "/long_samples"
    
    exp_meta_data = pd.read_csv(exp_dir + "/" + exp_num_str + "_meta_data.csv", index_col=[0])
    
    dfs_list = []
    
    for root, dirs, files in os.walk(long_samps_dir):
        for f in files:
            print(f)
            if f[-4:]==".csv":
                beta_name = f[:-9]
                print(beta_name)
                
                if not is_24:
                    if beta_name[-1] == "2":
                        
                        df = pd.read_csv(long_samps_dir + "/" + f, index_col=[0])
                        df_mean = df.groupby(['radius_label']).agg({'Ln_w': ['mean', 'std'], 'radius_label':'count'})
                        df_mean = df_mean.droplevel(axis=1, level=0).reset_index()
                        df_mean['symm_angle'] = df['symm_angle'].iloc[0]
                        # print(df_mean.index)
                        
                        dfs_list.append(df_mean)
                        
                else:
                    if beta_name[-1] == "0":
                        df = pd.read_csv(long_samps_dir + "/" + f, index_col=[0])
                        df_mean = df.groupby(['radius_label']).agg({'Ln_w': ['mean', 'std'], 'radius_label':'count'})
                        df_mean = df_mean.droplevel(axis=1, level=0).reset_index()
                        df_mean['symm_angle'] = df['symm_angle'].iloc[0]
                        # print(df_mean.index)
                        
                        dfs_list.append(df_mean)
            
    
    
    df_ct = pd.concat(dfs_list)
    df_ct_name = exp_dir + "/" + exp_num_str + "_radius.csv"
    df_ct.to_csv(df_ct_name, index=True)
    
    return df_ct

#%%

def radius_label_talls(epsilon_list, exp_nums):
    
    dfs_list = []
    for exp_num in exp_nums:
        exp_num_str = "EXP{:0>2}".format(exp_num)
        exp_dir = '/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/' + exp_num_str
        val_samps_dir = exp_dir + "/validated_samples"
        
        exp_meta_data = pd.read_csv(exp_dir + "/" + exp_num_str + "_meta_data.csv", index_col=[0])
        
        
        is_24 = exp_num == 24
        for root, dirs, files in os.walk(val_samps_dir):
            for f in files:
                if f[-4:]==".csv":
                    beta_name = f[:-4]
                    
                    if not is_24:
                        if beta_name[-1] == "2":
                            df = pd.read_csv(val_samps_dir + "/" + f, index_col=[0])
                            
                            for r in epsilon_list:
                                df['radius_label_{0:.2f}'.format(r)] = df.apply(lambda row: new_drivers.radius_label(row,r), axis=1 )
                            
                            
                            df_tall = tall_samples.pivot_long(df, [], [], num_hidden=2)
                            dfs_list.append(df_tall)
                            print(beta_name)
                            print(df_tall.columns)
                            print(df_tall.head())
                    else: 
                        if beta_name[-1] == "0":
                            df = pd.read_csv(val_samps_dir + "/" + f, index_col=[0])
                            
                            for r in epsilon_list:
                                df['radius_label_{0:.2f}'.format(r)] = df.apply(lambda row: new_drivers.radius_label(row,r), axis=1 )
                            
                            
                            df_tall = tall_samples.pivot_long(df, [], [], num_hidden=2)
                            dfs_list.append(df_tall)
                            
    big_df = pd.concat(dfs_list)
    
    big_df.to_csv('/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/' + "EXP2X_radius_tall.csv", index=True)
    
    
    
    # for exp_long in exp_long_list:
    #     exp_long_val = exp_long.replace('long_samples', 'validated_samples')
    #     exp_long_val = exp_long_val.replace('_long', '')
    #     exp_meta_data = pd.read_csv(exp_long + "/" + exp_num_str + "_meta_data.csv", index_col=[0])
        
        
    #     df = pd.read_csv(exp_long_val, index_col=[0])
        
    #     for r in epsilon_list:
    #         df['radius_label_{0:.2f}'.format(r)] = df.apply(lambda row: new_drivers.radius_label(row), axis=1, args=(r))
        
    #     df_tall = tall_samples.pivot_long(df,[],[], num_hidden=2)
        
        
        
        
    # pivot_long(samples, exp_meta_data, exp_code, normalise=True, true_nodes = False):
        
    # tall_samples.pivot_driver(directory_dict['meta_experiment_folder'], only_bin=False)    
    
    # exp_num_str = "EXP{:0>2}".format(exp_num)
    # exp_dir = '/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/' + exp_num_str
    # long_samps_dir = exp_dir + "/long_samples"
    
    # exp_meta_data = pd.read_csv(exp_dir + "/" + exp_num_str + "_meta_data.csv", index_col=[0])
    

epsilon_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
exp_nums = [20,21,22,23,24]
radius_label_talls(epsilon_list, exp_nums)




#%%

        
exp_long_list = ['/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP20/long_samples/EXP20P004B02_long.csv',
            '/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP21/long_samples/EXP21P003B02_long.csv',
            "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP22/long_samples/EXP22P000B02_long.csv",
            "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP22/long_samples/EXP22P002B02_long.csv",
            "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP22/long_samples/EXP22P004B02_long.csv",
            "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP23/long_samples/EXP23P001B02_long.csv",
            "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP24/long_samples/EXP24P000B00_long.csv",
            "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP24/long_samples/EXP24P002B00_long.csv"]
    






#%%

df_ct_list = []
for en in [20,21,22,23,24]:
    
    df_ct = radius_PT_df_generator(en, en==24)
    df_ct_list.append(df_ct)
    
df_final = pd.concat(df_ct_list)

df_final_no_outlier = df_final.loc[df_final['radius_label']!= "outlier"]
sns.lineplot(data = df_final_no_outlier, x="symm_angle", y="mean", hue='radius_label')

# %%

    
df = pd.read_csv("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP24/long_samples/EXP24P002B00_long.csv", index_col=[0])

sns.scatterplot(data=df, x="w_i2", y="Ln_w", hue="radius_label")
plt.ylim([1.411,1.413])    
    
    
    
#%%

df = pd.read_csv("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP24/long_samples/EXP24P002B00_long.csv", index_col=[0])

sns.scatterplot(data=df, x="w_i1", y="w_i2", hue="radius_label", palette="viridis", s=10)
plt.ylim([-0.5,2.5])
plt.xlim([-1,1])
plt.savefig("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/pi.2_scatter_radius.png")



#%%

df = pd.read_csv("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP24/long_samples/EXP24P002B00_long.csv", index_col=[0])
df = df.loc[df['radius_label']!='outlier']


cmap = sns.color_palette("magma", as_cmap=True)

#points = plt.scatter(x=df['w_1_norm'], y=df['w_2_norm'], c=df['free_energy_cluster_average'], s=10, cmap=cmap)

sns.scatterplot(x=df["w_i1"], y=df["w_i2"], c=df["Ln_w"], cmap=cmap, s=2)
plt.ylim([-0.5,2.5])
plt.xlim([-1,1])
folder = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/"
plt.savefig(folder + "pi.2_scatter_Ln_w.png")




#%%


df_big = pd.read_csv("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP2X_radius_tall.csv", index_col=[0])


#%%

#df=df_big.sample(frac=0.2, replace=False)
df = df_big

epsilon_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

for eps in epsilon_list:
    rl = 'radius_label_{0:.2f}'.format(eps)
    
    df_mean = df.groupby([rl, 'symm_angle']).agg({'Ln_w': ['mean', 'std'], rl:'count'})
    df_mean = df_mean.droplevel(axis=1, level=0).reset_index()
    
    print(df_mean)
    
    df_final_no_outlier = df_mean.loc[df_mean[rl]!= "outlier"]
    
    # plt.clf()
    # sns.lineplot(data = df_final_no_outlier, x="symm_angle", y="mean", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_mean.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_final_no_outlier, x="symm_angle", y="std", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_std.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_final_no_outlier, x="symm_angle", y="count", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_count.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_mean, x="symm_angle", y="mean", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_mean.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_mean, x="symm_angle", y="std", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_std.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_mean, x="symm_angle", y="count", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_count.png".format(eps))

    plt.clf()
    # df_log = df[['symm_angle', 'Ln_w', rl]]
    # df_log['Ln_w'] = np.log(df_log['Ln_w'])
    df=df.sort_values(by=[rl])
    
    # relabelling_dict = {'degen': r'$\mathcal{R}^1_{\varepsilon}$', 
    #                     'non_degen': r'$\mathcal{R}^0_{\varepsilon}$', 
    #                     'outlier': r'$W \, \backslash \, (\mathcal{R}^1_{\varepsilon} \cup \mathcal{R}^0_{\varepsilon})$'}
    
    relabelling_dict = {'degen': r'$\mathcal{A}(0,\varepsilon) \sqcup \mathcal{A}(2, \varepsilon) $', 
                        'non_degen': r'$\mathcal{A}(1,\varepsilon)^2$', 
                        'outlier': r'$\mathcal{W} \, ^c$'}
    
    
    
    df[rl] = df[rl].replace(relabelling_dict)
    df_mean[rl] = df_mean[rl].replace(relabelling_dict)
    
    #print(df.head())
    
    sns.lineplot(data = df.loc[df[rl]!= relabelling_dict['outlier']] ,
                 x="symm_angle", y="Ln_w", hue=rl, ci='sd', 
                 palette='magma', err_style='band', markers=True, 
                 #err_kws={'elinewidth':0.5}
                 )
    
    sns.lineplot(data = df_mean.loc[df_mean[rl]==relabelling_dict['outlier']], x='symm_angle', y='mean', 
                 palette = 'magma', linestyle='--',
                 color = 'green', label = relabelling_dict['outlier'])
    plt.legend(title='Region of $W$', loc='best')

    plt.title(r"Free energy with $\varepsilon = $ {}".format(eps))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\overline{F}_n(\mathcal{W})$')
    
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig("FE_radius_{}_symm_angle.png".format(eps))

    
    # plt.clf()
    # sns.lineplot(data = df, x="symm_angle", y="count", hue=rl)
    # plt.savefig("B_radius_{}_symm_angle_Ln_w_count.png".format(eps))


#%%

df = df_big

epsilon_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

for eps in epsilon_list:
    rl = 'radius_label_{0:.2f}'.format(eps)
    df_mean['symm_angle'].round(2)
    
    df_mean = df.groupby([rl, 'symm_angle']).agg({'Ln_w': ['mean', 'std'], rl:'count'})
    df_mean = df_mean.droplevel(axis=1, level=0).reset_index()

    
    total_dict = {angle : df_mean.loc[df_mean['symm_angle']==angle]['count'].sum() for angle in np.unique(df_mean['symm_angle'])}
    
    df_mean['prop'] = df_mean.apply(lambda row: row['count']/total_dict[row['symm_angle']], axis=1)

    
    plt.clf()
    df=df.sort_values(by=[rl])
    
    relabelling_dict = {'degen': r'$\mathcal{A}(0,\varepsilon) \sqcup \mathcal{A}(2, \varepsilon) $', 
                        'non_degen': r'$\mathcal{A}(1,\varepsilon)^2$', 
                        'outlier': r'$\mathcal{W}^c$'}
    
    
    
    df[rl] = df[rl].replace(relabelling_dict)
    df_mean[rl] = df_mean[rl].replace(relabelling_dict)
    
    #print(df.head())
    
    sns.lineplot(data = df_mean.loc[df_mean[rl]!= relabelling_dict['outlier']] ,
                 x="symm_angle", y="prop", hue=rl,
                 palette='magma', 
                 #err_kws={'elinewidth':0.5}
                 )
    
    sns.lineplot(data = df_mean.loc[df_mean[rl]==relabelling_dict['outlier']], x='symm_angle', y='prop', 
                 palette = 'magma', linestyle='--',
                 color = 'green', label = relabelling_dict['outlier'])
    plt.legend(title='Region of $W$', loc='best')

    plt.title(r"Density of regions with $\varepsilon = $ {}".format(eps))
    plt.xlabel(r'$\theta$')
    plt.ylabel('Density')
    
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig("radius_{}_symm_angle_density_counts.png".format(eps))
    





#%% FREE ENERGY STUFF

def get_true_Ln(exp_num_dict):
    true_paths=[]
    for en in exp_num_dict:
        meta_df = pd.read_csv("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP{}/EXP{}_meta_data.csv".format(en, en), index_col=[0])
        
        for P in range(exp_num_dict[en]['Prange']):
            for T in range(exp_num_dict[en]['Trange']):
                exp_name = "EXP{}P00{}B0{}T0{}".format(en, P, exp_num_dict[en]['beta'], T)
                z_score = float(meta_df.loc[meta_df['exp_code']==exp_name, 'Ln_w_z_score'])
                if np.abs(z_score)<1.5:
                    true_paths.append("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP{}/true_parameters/{}_w0.csv".format(en,exp_name))
    
    
    df_true = pd.concat([pd.read_csv(x, index_col=[0]) for x in true_paths])

    return df_true


exp_num_dict = {20: {'Prange':5, 'Trange':8, 'beta':2},
                21: {'Prange':5, 'Trange':8, 'beta':2},
                22: {'Prange':5, 'Trange':8, 'beta':2},
                23: {'Prange':4, 'Trange':8, 'beta':0},
                24: {'Prange':3, 'Trange':8, 'beta':0},
                }



def agg_df(df,eps, true_param=False):
    
    rl = 'radius_label_{0:.2f}'.format(eps)
    print(rl)
    if not true_param:
        df_mean = df.groupby([rl, 'symm_angle']).agg({'Ln_w': ['mean', 'std', 'min'], rl:'count'})
    else:
        df_mean = df.groupby(['symm_angle']).agg({'Ln_w': ['mean', 'std', 'min']})
    df_mean = df_mean.droplevel(axis=1, level=0).reset_index()
    n=10000
    df_mean['rlct'] = df_mean.apply(lambda row: n*(row['mean']-row['min'])/np.log(n) , axis=1)

    return df_mean

#%%

df = df_big
df_true = get_true_Ln(exp_num_dict)

epsilon_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

for eps in epsilon_list:
    
    rl = 'radius_label_{0:.2f}'.format(eps)
    df_mean = agg_df(df,eps)
    df_mean_true = agg_df(df_true,eps,True)
    
    print(df_mean)
    
    df_final_no_outlier = df_mean.loc[df_mean[rl]!= "outlier"]
    
    total_dict = {angle : df_mean.loc[df_mean['symm_angle']==angle]['count'].sum() for angle in np.unique(df_mean['symm_angle'])}
    df_mean['prop'] = df_mean.apply(lambda row: row['count']/total_dict[row['symm_angle']], axis=1)

    
    # plt.clf()
    # sns.lineplot(data = df_final_no_outlier, x="symm_angle", y="mean", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_mean.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_final_no_outlier, x="symm_angle", y="std", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_std.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_final_no_outlier, x="symm_angle", y="count", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_count.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_mean, x="symm_angle", y="mean", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_mean.png".format(eps))
    
    # plt.clf()
    # sns.lineplot(data = df_mean, x="symm_angle", y="std", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_std.png".format(eps))
    
    
    #print(df.head())

    
    # sns.lineplot(data = df.loc[df[rl]!= relabelling_dict['outlier']] ,
    #              x="symm_angle", y="Ln_w", hue=rl, ci='sd', 
    #              palette='magma', err_style='band', markers=True, 
    #              #err_kws={'elinewidth':0.5}
    #              )
    
    # sns.lineplot(data = df_mean.loc[df_mean[rl]==relabelling_dict['outlier']], x='symm_angle', y='mean', 
    #              palette = 'magma', linestyle='--',
    #              color = 'green', label = relabelling_dict['outlier'])
    
    
    # plt.clf()
    # sns.lineplot(data = df_mean, x="symm_angle", y="min", hue=rl)
    # plt.savefig("radius_{}_symm_angle_Ln_w_count.png".format(eps))
    
    

    plt.clf()
    df=df.sort_values(by=[rl])

    relabelling_dict = {'degen': r'$\mathcal{A}_{\mathrm{Degen}}$', 
                        'non_degen': r'$\mathcal{A}_{\mathrm{NonDegen}}$', 
                        'outlier': r'$\mathcal{A} \, ^c$'}
    
    
    
    df[rl] = df[rl].replace(relabelling_dict)
    df_mean[rl] = df_mean[rl].replace(relabelling_dict)
    
    
    
    palette = {relabelling_dict['degen']:"#440154FF", relabelling_dict['non_degen']: "#FDE725FF"}
    
    plt.clf()
    sns.lineplot(data = df_mean.loc[df_mean[rl]!= relabelling_dict['outlier']], 
                 x="symm_angle", y="min", hue=rl, palette=palette, markers=True)
    sns.lineplot(data = df_mean.loc[df_mean[rl]== relabelling_dict['outlier']], 
                 x="symm_angle", y="min", color='green', linestyle='--',
                 label = relabelling_dict['outlier'],
                 markers=True)
    sns.scatterplot(data = df_mean_true, x='symm_angle', y='min',color='red', label='Truth')
    
    
    
    plt.legend(title='Region of $W$', loc='best')

    plt.title(r"$L_n(\omega_0)$ with $\varepsilon = $ {}".format(eps))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$L_n(\omega_0)$')
    
    plt.tight_layout()
    plt.savefig("Ln_min_radius_{}_symm_angle.png".format(eps))
    
    
    
    
    
    
    
    plt.clf()
    sns.lineplot(data = df_mean.loc[df_mean[rl]!= relabelling_dict['outlier']], 
                 x="symm_angle", y="rlct", hue=rl, palette=palette, markers=True)
    sns.lineplot(data = df_mean.loc[df_mean[rl]== relabelling_dict['outlier']], 
                 x="symm_angle", y="rlct", color='green', linestyle='--',
                 label = relabelling_dict['outlier'],
                 markers=True)
    #sns.scatterplot(data = df_mean_true, x='symm_angle', y='rlct')
    
    
    
    plt.legend(title='Region of $W$', loc='best')

    plt.title(r"$\lambda$ with $\varepsilon = $ {}".format(eps))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\lambda$')
    
    plt.tight_layout()
    plt.savefig("RLCT_radius_{}_symm_angle.png".format(eps))
    
    
    
    
    
    plt.clf()
    sns.lineplot(data = df_mean.loc[df_mean[rl]!= relabelling_dict['outlier']] ,
                 x="symm_angle", y="prop", hue=rl,
                 palette=palette, 
                 #err_kws={'elinewidth':0.5}
                 )
    
    sns.lineplot(data = df_mean.loc[df_mean[rl]==relabelling_dict['outlier']], x='symm_angle', y='prop', 
                 linestyle='--',
                 color = 'green', label = relabelling_dict['outlier'])
    plt.legend(title='Region of $W$', loc='best')

    plt.title(r"Density of regions with $\varepsilon = $ {}".format(eps))
    plt.xlabel(r'$\theta$')
    plt.ylabel('Density')
    
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig("radius_{}_symm_angle_density_counts.png".format(eps))
    
    
    
    
#%%


plt.clf()
sns.scatterplot(data = df_true, x='symm_angle', y='Ln_w')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$S_n$')
plt.title(r'$S_n$ for each trial')

plt.tight_layout()
plt.savefig("S_n_vs_theta.png")








    
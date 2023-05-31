import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
import pandas as pd
import numpy as np
import tall_samples
from scipy import interpolate

# PLOT 1 

def scatterplot(df_long, exp_meta_data, true_samples, exp_code, exp_plot_path, 
                lims, title=False, subtitle=False, hat=True):
    
    # true_samples = tall_samples.pivot_long(true_samples, exp_meta_data, exp_code, normalise=True)
    
    fig, ax = plt.subplots(1,1)
    ax.cla() # almost certainly don't actually need this
    
    sns.scatterplot(data = df_long, x='w_i1', y='w_i2', ax=ax, hue='index', s=10, palette="viridis")
    ax.plot(true_samples['w_i1'], true_samples['w_i2'], 'o', color='red')
    #ax.axis('equal')
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    if title:
        ax.set_title(title)
        
    if hat:
        ax.set_xlabel(r'$\hat{w}_{i,1}$')
        ax.set_ylabel(r'$\hat{w}_{i,2}$')
    else:
        ax.set_xlabel(r'$w_{i,1}$')
        ax.set_ylabel(r'$w_{i,2}$')

    fig.tight_layout()
    plt.close()
    
    fig.savefig(exp_plot_path)
    
def single_density_plot(df_long, exp_meta_data, true_samples, exp_code, 
                        exp_plot_path, lims=([-2,2],[-2,2]), lims_true=False, title=False, subtitle=False, ):
    
    
    df_long = df_long.round({'symm_angle': 2})
    true_samples = true_samples.round({'symm_angle':2})
    
    df_long = df_long.rename(columns = {"symm_angle":'theta'})
    true_samples = true_samples.rename(columns = {"symm_angle":'theta'})

    fig, ax = plt.subplots(1,1)
    ax.cla()
    sns.kdeplot(data=df_long, x = "w_i1", y = "w_i2", ax=ax, clip=([-5,5], [-5,5]))
    ax.plot(true_samples['w_i1'], true_samples['w_i2'], 'o', color='red')
    #ax.axis('equal')
    if lims_true:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    # if title:
    #     ax.set_title(title)
        
    symm_angle = df_long['theta'].iloc[0]
    ax.set_title(r'$\theta$={}'.format(symm_angle))
    ax.set_xlabel(r'$\hat{w}_{i,1}$')
    ax.set_ylabel(r'$\hat{w}_{i,2}$')
    
    #g.set_axis_labels(x_var=r'$\hat{w}_{i,1}$', y_var=r'$\hat{w}_{i,2}$')

    fig.tight_layout()
    plt.close()
    
    fig.savefig(exp_plot_path)
    
    
    
    
def free_energy_plot(df_long, df_V_long, exp_meta_data, true_samples, exp_code, 
                        exp_plot_path, lims, V_type, regression=False, title=False, subtitle=False, ):
    
    # V_type is a string either 'V_angle' or 'V_new'
    
    fig, ax = plt.subplots(1,1)
    ax.cla()
    sns.scatterplot(data = df_long, x=  V_type, y='Ln_w', ax=ax, hue='index_eps', s=10, palette="viridis")
    ax.plot(true_samples[V_type], true_samples['Ln_w'], 'o', color='red')
    
    if regression:
        sns.regplot(df_long, x=V_type, y='Ln_w', order=4)
    else:
        sns.lineplot(data=df_V_long, x=V_type, y='Ln_w')
    
    ax.axis('equal')
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    if title:
        ax.set_title(title)

    fig.tight_layout()
    plt.close()
    
    fig.savefig(exp_plot_path)
    
def free_energy_combined(df_long, df_V_long, exp_meta_data, true_samples, exp_code, 
                        exp_plot_path, lims, V_type, order_param, regression=False, title=False, subtitle=False, ):
    
    # df_long and df_V_long should have been compiled to contain all parameters 
    # order_param is a string either 'eps_angle', 'x_max', 'symm_angle'
    
    fig, ax = plt.subplots(1,1)
    ax.cla()
    
    sns.lineplot(data=df_V_long, x=V_type, y='Ln_w', ax=ax, hue=order_param, palette = "viridis")
    sns.scatterplot(data=true_samples, x=V_type, y='Ln_w', ax=ax, hue=order_param, palette = "viridis")
    
    
    
    
    
    
    
    
    
    
    
        

    

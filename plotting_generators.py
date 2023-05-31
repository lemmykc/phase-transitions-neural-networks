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

# plot_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/Plots"
# exp_name = "EXP09"

# plot_dir = new_drivers.directory_creator(plot_path, exp_name)



# ### Normalised and unnormalised scatters for trial 

# trial_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP05/long_samples/EXP05P000B00_long.csv"
# trial_meta_data_path = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP05/EXP05_meta_data.csv"

def get_dfs(exp_dir, beta_name):
    meta_exp_name = exp_dir[-5:]
    df_long = pd.read_csv(exp_dir + "/long_samples/" + beta_name + "_long.csv", index_col=[0])
    exp_meta_data = pd.read_csv(exp_dir + "/" + meta_exp_name + "_meta_data.csv")
    true_samples = pd.read_csv(exp_dir + "/true_parameters/" + beta_name + "T00_w0.csv")
    true_samples_long = tall_samples.pivot_long(true_samples, exp_meta_data, beta_name, normalise=True, true_nodes=True)
    
    return df_long, exp_meta_data, true_samples_long

def normal_unnormal(exp_dir, beta_name, plot_path, lims = [(-3, 3), (-3,3)]):
    
    
    df_long, exp_meta_data, true_samples = get_dfs(exp_dir, beta_name)
    
    exp_plot_dir = new_drivers.directory_creator(plot_path, exp_dir[-5:])
    exp_plot_dir_norm = new_drivers.directory_creator(exp_plot_dir, "scatter_norm")
    exp_plot_dir_unnorm = new_drivers.directory_creator(exp_plot_dir, "scatter_unnorm")
    
    
    # Get normalised
    exp_plot_path = exp_plot_dir_norm + "/" + beta_name + "_scatter_norm.png"
    plotting.scatterplot(df_long, exp_meta_data, true_samples, beta_name, exp_plot_path, lims = lims)
    
    
    
    # Get unnormalised
    exp_plot_path = exp_plot_dir_unnorm + "/" + beta_name + "_scatter_unnorm.png"
    samples = pd.read_csv(exp_dir + "/validated_samples/" + beta_name + ".csv")
    df_unn_long = tall_samples.pivot_long(samples, exp_meta_data, beta_name, normalise=False)
    plotting.scatterplot(df_unn_long, exp_meta_data, true_samples, beta_name, exp_plot_path, lims=lims)
    



def single_density(exp_dir, beta_name, plot_path, lims = [(-3, 3), (-3,3)], create_folder=True):
    
    
    df_long, exp_meta_data, true_samples = get_dfs(exp_dir, beta_name)
    
    if create_folder:
        exp_plot_dir = new_drivers.directory_creator(plot_path, beta_name[:5])
        exp_plot_dir_den = new_drivers.directory_creator(exp_plot_dir, "density_norm")
    else:
        exp_plot_dir_den = plot_path 
    # Get density
    
    exp_plot_path =exp_plot_dir_den + "/" + beta_name + "_density.png" 
    plotting.single_density_plot(df_long, exp_meta_data, true_samples, beta_name, exp_plot_path, lims=lims, lims_true=True)



def plot_all_beta(meta_dir, lims=[(-0.75,0.75), (-1,3)], plot_dir = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/Plots"):
    val_dir = meta_dir + "/validated_samples"
    for root, dirs, files in os.walk(val_dir):
        for f in files:
            beta_name = f[:-4]
            single_density(meta_dir, beta_name, plot_dir, lims )
                #plotting_generators.normal_unnormal(directory_dict['meta_experiment_folder'], beta_name, plot_dir, )


#meta_dir = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP21"




import os
import tall_samples
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})


raw_samps = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP16/raw_samples"
exp_dir = "/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP16"
exp_meta_data = pd.read_csv("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP16/EXP16_meta_data.csv", index_col=[0])


for root, dirs, files in os.walk(raw_samps):
    for f in files:
    
        beta_name = f[:-7]
        samples = pd.read_csv(os.path.join(root, f), index_col = [0])
        
        df_long = tall_samples.pivot_long(samples, [],[], num_hidden=3)
        
        true_samples = pd.read_csv(exp_dir + "/true_parameters/" + beta_name + "T00_w0.csv")
        true_samples_long = tall_samples.pivot_long(true_samples, exp_meta_data, beta_name, normalise=True, true_nodes=True)
        
        
        fig, ax = plt.subplots(1,1)
        ax.cla()
        sns.kdeplot(data=df_long, x = "w_i1", y = "w_i2", ax=ax, clip=([-4,2], [-2,2]))
        ax.plot(true_samples_long['w_i1'], true_samples_long['w_i2'], 'o', color='red')
        

        fig.tight_layout()
        plt.close()
        
        fig.savefig(exp_dir + "/" + f[:-4] + ".png")
#%%
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
sns.set(rc={"figure.dpi":500, 'savefig.dpi':500})
matplotlib.rcParams.update({'font.size': 3, 'legend.fontsize': 10, 
                            'legend.title_fontsize':10,
                            'legend.loc':'upper right'})
plt.rc('xtick', labelsize=5)  
plt.rc('ytick', labelsize=5)  
# plt.rc('xlabel', labelsize=5)  
# plt.rc('ylabel', labelsize=5)  
import tall_samples

from sklearn import mixture


#%%
df = pd.read_csv("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP13/raw_samples/EXP13P000B00T00.csv")

df_long_norm = tall_samples.pivot_long(df, [], [], num_hidden=3, normalise=True)
df_long_norm['norm']=True
df_long_unnorm = tall_samples.pivot_long(df, [], [], num_hidden=3, normalise=False)
df_long_unnorm['norm']=False

df_long = pd.concat([df_long_norm, df_long_unnorm])


df_true = pd.read_csv("/Users/liam/Documents/Uni Stuff/Deep Learning Masters Research/Semester 3/Final_Experiments/EXP13/true_parameters/EXP13P000B00T00_w0.csv", index_col=[0])
true_samples = tall_samples.pivot_long(df_true, [], [], num_hidden=3, normalise = True, true_nodes=True)




#%% 

#sns.kdeplot(data=df_long_norm, x='w_i1', y='w_i2', hue='angle_label', palette='viridis')

# X_train = df_long_norm[['w_i1', 'w_i2', 'eps_sign']].to_numpy()
# means_init = np.vstack([[np.cos(t*np.pi/3), np.sin(t*np.pi/3), 2*(t % 2)-1] for t in range(6)])

# X_train = df_long_norm[['w_i1', 'w_i2']].to_numpy()
# means_init = np.vstack([[np.cos(t*np.pi/3), np.sin(t*np.pi/3)] for t in range(6)])


X_train = df_long_norm[['w_i1', 'w_i2', 'angle_label']].to_numpy()
df_long_norm['angle_label'].replace({'eps':-1, 'not_eps':1 }, inplace=True)
means_init = np.vstack([[np.cos(t*np.pi/3), np.sin(t*np.pi/3), 2*(t % 2)-1] for t in range(6)])
#means_init = np.c_[means_init, np.tile(['eps', 'not_eps'],3)]


clf = mixture.GaussianMixture(n_components=6, covariance_type='full', means_init=means_init)
clf.fit(X_train)



df_long_norm['mixture_comp'] = clf.predict(X_train)

df_long_norm['mixture_comp_mod2'] = df_long_norm['mixture_comp'].to_numpy() %2

# df_long_norm['mixture_comp'] = df_long_norm.apply(lambda row: clf.fit(row[['w_i1', 'w_i2']].to_numpy().reshape(1,-1)), axis=1)

sns.kdeplot(data=df_long_norm, x='w_i1', y='w_i2', hue='mixture_comp_mod2', palette='viridis', thresh=0.1)
#sns.scatterplot(data=df_long_norm, x='w_i1', y='w_i2', hue='mixture_comp', palette='viridis')


# df['angle_label'] = df.apply(lambda row: angle_eps(row['V_angle']), axis=1)



#%%
# SCATTERPLOT
#plt.rc('figure', figsize=(16, 9))
g = sns.FacetGrid(df_long, col="norm", height=2.95, aspect = 1)
g.map_dataframe(sns.scatterplot, x="w_i1", y="w_i2", hue='index', s=2, palette='viridis')

# plt.gca().set_aspect(0.5)

axes = g.axes.flatten()

for ax in axes:
    ax.plot(true_samples['w_i1'], true_samples['w_i2'], 'o', color='red')
    ax.set_title('')
    
# g.set_yticklabels(size = 5)
# g.set_xticklabels(size = 5)

#ax.axis('equal')
ax.set_xlim([-2.5,2.5])
ax.set_ylim([-2,2])

g.axes[0,0].set_xlabel(r'$w_{i,1}$', fontsize=10)
g.axes[0,1].set_xlabel(r'$\hat{w}_{i,1}$', fontsize = 10)
g.axes[0,0].set_ylabel(r'$w_{i,2}$', fontsize = 10)
g.axes[0,1].set_ylabel(r'$\hat{w}_{i,2}$', fontsize = 10)

# g.set_axis_labels(x_var=r'$\hat{w}_{i,1}$', y_var=r'$\hat{w}_{i,2}$', fontsize=18)
g.add_legend(title='Index', prop={'size': 6}, loc=1)
#plt.legend(prop={'size': 6})

g.tight_layout() 
#plt.subplots_adjust(hspace = 2)
g.savefig("scatter_facet.png")

#%%

fig, ax = plt.subplots(1,1)
ax.cla() # almost certainly don't actually need this

#plt.figure(figsize=(16, 9))

sns.scatterplot(data = df_long, x='w_i1', y='w_i2', ax=ax, hue='index_eps', s=10, palette="viridis")
ax.plot(true_samples['w_i1'], true_samples['w_i2'], 'o', color='red')
#ax.axis('equal')
ax.set_xlim([-2.5,2.5])
ax.set_ylim([-2,2])
ax.set_xlabel(r'$w_{i,1}$')
ax.set_ylabel(r'$w_{i,2}$')


fig.tight_layout()
plt.close()

fig.savefig("scatter_test.png")



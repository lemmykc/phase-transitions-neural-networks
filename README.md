# Phase Transitions in Two Layer Feedforward ReLU Neural Networks

This code was written for my (Liam Carroll) Master's Thesis on [_Phase Transitions in Neural Networks_](http://therisingsea.org/notes/MSc-Carroll.pdf) under the supervision of Dr. Daniel Murfet at the University of Melbourne. The initial template for this code is thanks to Susan Wei, Daniel Murfet et al. from their work on [Deep Learning is Singular, and That's Good](https://arxiv.org/abs/2010.11560). 

I am uploading this code as a template for others that wish to perform similar experiments measuring phase transitions in the Bayesian posterior, in the knowledge that it is certainly not clean enough to be a package, but warrants existence in the ether nonetheless. 

Half of the files are reasonably clean and easy to follow. The other half, mainly the ones for plotting the posterior and related results, are excrutiatingly hard to read, but that is because I simply wrote them for myself with no other user in mind. Some of the plotting techniques may be useful for a new user, but it is likely going to be quicker to write your own plots than to depend on how I structured mine. (A good lesson for myself in future theses).

I imagine future users will use these experiments as inspiration, use the bones of the MCMC process here to study different kinds of singular models and true distributions, and then perform their own data analysis specific to the problem at hand. In any case, I hope you find the code a little bit useful. 

# Running the experiments
The following files are the key ones in running the experiments to attain samples from a posterior 

### `driver_script_new`
Where to define experiment metadata like hyperparameters and other settings, and run the experiments in parallel.

### `classes`
Where a single experiment's hyperparameters are initialised, and the posterior sampling driver is defined. 

### `model_setup` 
Where the true network, true dataset $D_n$, and the pyro neural network model are defined. 

### `HMC_inference`
Where the HMC NUTS variant of MCMC is run, producing samples from the posterior. 

# Data management
These files are largely boring but are somewhat necessary in making the experiment code above not break. 

### `data_collector`
Calculates the negative log loss $L_n(w)$ on each sample and collects all samples into a useable dataframe for further analysis. This dataframe is flat, not tall. 

### `tall_samples`
Get samples_df into tall format for plotting. 

### `new_drivers`
An assortment of data mangement functions (like defining new folders etc.), and a post-production file for performing statistical validation. Also runs driver to pivot validated samples into long format dataframe for plotting purposes (where each sample is represented $d$ times due to the permutation symmetry). 

# Assorted other files specific to my particular experiments

### `radius_generator`
Calculate and plot samples from each annuli. Relevant to PT1 "Deforming to Degeneracy" in the thesis. 

### `RLCT_estimate`
An attempt to calculate and plot the RLCT. This ultimately did not prove fruitful for me, but maybe a future experimenter will have more luck. 

# Messy plotting files
The key plots of my thesis were born somewhere among the rest of the files in this folder. If you are just looking to perform some similar MCMC experiments, I would recommend not touching these for a while and just focusing on the first four files. 

If you have questions about anything, feel free to reach out at 
lemmykc at gmail dot com
 



import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

import bayesiantests as bt

import matplotlib.pyplot as plt
import seaborn as snb
import scipy.io as sio


# Define set that stores names of results files.
res_list = []

# Define path to folder containing results.
RESULTS_FOLDER = '../results-single-dataset/'
RESULTS_FOLDER_PLOTS = '../results-single-dataset/plots/'

# Go over results files in results folder.
for file_name in os.listdir(RESULTS_FOLDER):
    if file_name.split('.')[-1] == 'mat':
        res_list.append(file_name)

# Set rope values
rope = 0.01

# Go over pairs of algorithms and compare.
for idx1 in np.arange(len(res_list)-1):
    for idx2 in np.arange(idx1+1, len(res_list)):
        
        # Load results.
        acc1 = np.ravel(sio.loadmat(RESULTS_FOLDER + res_list[idx1])['res'])     # pleft
        acc2 = np.ravel(sio.loadmat(RESULTS_FOLDER + res_list[idx2])['res'])     # pright

        # Specify names.
        names = (res_list[idx1].split(".")[0].split("_")[1], res_list[idx2].split(".")[0].split("_")[1])

        # Difference of accuracies.
        x = acc1 - acc2

        # Perform Bayesian correlated t-test.
        pleft, prope, pright = bt.correlated_ttest(x, rope=rope, runs=10, verbose=True, names=names)

        # Save results to file.
        with open('../results-single-dataset/results_single_dataset.res', 'a') as f:
            f.write('{0} - {1}: pleft={2}, prope={3}, pright={4}\n'.format(names[0], names[1], pleft, prope, pright))
        
        ### RESULTS PLOT ###

        # Generate samples from posterior (it is not necesssary because the posterior is a Student).
        samples=bt.correlated_ttest_MC(x, rope=rope, runs=10, nsamples=50000)

        # Plot posterior.
        snb.kdeplot(np.array(samples), shade=True)

        # Plot rope region.
        plt.axvline(x=-rope,color='orange')
        plt.axvline(x=rope,color='orange')

        # Add label.
        plt.xlabel(names[0] + " - " + names[1])

        # Save figure.
        plt.savefig(RESULTS_FOLDER_PLOTS + names[0] + '_' + names[1] + '.png')
        plt.clf()

        ####################



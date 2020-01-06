import os
import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import bayesiantests as bt


def construct_scores_matrix(alg1, alg2):
    """Construct scores matrix for pair of compared RBAs.
    
    Args:
        alg1 (str): name of first RBA.
        alg2 (str): name of second RBA.

    Returns:
        (numpy.ndarray): matrix containing the CV fold score differences for pair
        of compared RBAs for different datasets.
    """

    # Get mapping of .mat files to contents.
    mapping_cont = dict()
    get_row_length = True
    row_length = None
    for f in filter(lambda x: x[0] != '.', os.listdir('../results-matrices/')):
        mapping_cont[f] = np.ravel(sio.loadmat('../results-matrices/' + f)['res'])
        if get_row_length:
            row_length = mapping_cont[f].shape[0]
            get_row_length = False

    
    # Get mapping of tuples of dataset names and algorithm names to CV fold scores.
    mapping_cont_new = dict()
    for key in mapping_cont.keys():
        spl = key.split('_')
        mapping_cont_new[(spl[0], spl[1].split('.')[0])] = mapping_cont[key]
    
    # Get set of all datset names for which results are available.
    dataset_names = set(map(lambda x: x[0], mapping_cont_new.keys()))

    # Allocate matrix for CV fold scores for each dataset.
    scores_mat = np.empty((len(dataset_names), row_length))

    # Go over datasets.
    idx = 0
    for dataset_name in dataset_names:
        # If there exist scores for both compared algorithms for next dataset, compute
        # difference and add to scores matrix.
        if (dataset_name, alg1) in mapping_cont_new and (dataset_name, alg2) in mapping_cont_new:
            res1 = mapping_cont_new[(dataset_name, alg1)]
            res2 = mapping_cont_new[(dataset_name, alg2)]
            scores_mat[idx, :] = res1 - res2
            idx += 1
    
    return scores_mat[:idx, :]


# Set rope and rho values.
rope=0.01
rho=1.0/10.0

# Parse compared algorithm names from command line arguments.
names = (sys.argv[1], sys.argv[2])

scores = construct_scores_matrix(*names)
msk = np.logical_not(np.apply_along_axis(lambda x: np.all(x == 0), 1, scores))
scores = scores[msk, :]

# Compute probabilities and write results to file.
pleft, prope, pright = bt.hierarchical(scores, rope, rho)
with open('../results-multiple-datasets/results_multiple_datasets.res', 'a') as f:
    f.write('{0} - {1}: pleft={2}, prope={3}, pright={4}\n'.format(names[0], names[1], pleft, prope, pright))

# Sample posterior, make simplex plot and save figure.
samples=bt.hierarchical_MC(scores, rope, rho, names=names)
fig = bt.plot_posterior(samples, names)
plt.savefig('../results-multiple-datasets/' + names[0] + '_' + names[1] + '.png')
plt.clf()


import numpy as np
import scipy.io as sio
import time

### INITIALIZATION ###

# Compare skrelief and skrebate implementation of ReliefF algorithm.
from skrelief.relieff import Relieff
from skrebate import ReliefF
# Set name of RBA implementation being evaluated.
rba_name = "ReliefF"

rba1 = Relieff()  # skrelief implementation of ReliefF algorithm.
rba2 = ReliefF(n_neighbors=5, n_jobs=1)   # SKRebate implementation of ReliefF algorithm.
rba3 = ReliefF(n_neighbors=5, n_jobs=-1)  # SKRebate implementation of ReliefF algorithm.

# Fit skrelief implementation of ReliefF algorithm to compile modules.
init_data = np.random.rand(30, 100)
init_target = (init_data[:, 0] > 0.5).astype(int)
rba1.fit(init_data, init_target)

RESULTS_SAVE_PATH = 'results/'

### /INITIALIZATION ###


# Create test data size parameters.
samples = np.arange(100, 101)
sample_dim = np.arange(100, 5100, 100)
xx, yy = np.meshgrid(samples, sample_dim)

# Get number of rows of data dimensionalities to compare.
num_pairs = xx.shape[0]

# Allocate matrices for time, were nxt_row are stored.
time_mat_1 = np.empty((sample_dim.size, samples.size), dtype=np.float)
time_mat_2 = np.empty((sample_dim.size, samples.size), dtype=np.float)
time_mat_3 = np.empty((sample_dim.size, samples.size), dtype=np.float)

# pairs indices.
idx_pair = 0

# Go over sizes of data matrices.
for pair in zip(xx, yy):
    
    # Allocate rows 
    nxt_row1 = np.empty(pair[0].size, dtype=np.float)
    nxt_row2 = np.empty(pair[0].size, dtype=np.float)
    nxt_row3 = np.empty(pair[0].size, dtype=np.float)
    
    # column index in next row of results.
    col_idx = 0
    for pair2 in zip(pair[0], pair[1]):

        # Create dummy data and target values.
        data = np.random.rand(pair2[0], pair2[1])
        target = (data[:, 0] > 0.5).astype(int)
        
        ### Measure performance ###
        start1 = time.perf_counter()
        rba1.fit(data, target)
        time1 = time.perf_counter() - start1

        start2 = time.perf_counter()
        rba2.fit(data, target)
        time2 = time.perf_counter() - start2

        start3 = time.perf_counter()
        rba3.fit(data, target)
        time3 = time.perf_counter() - start3
        ###########################
        
        # Save results to row of results.
        nxt_row1[col_idx] = time1
        nxt_row2[col_idx] = time2
        nxt_row3[col_idx] = time3
        col_idx += 1

    # Put rows in results matrices.
    time_mat_1[idx_pair, :] = nxt_row1
    time_mat_2[idx_pair, :] = nxt_row2
    time_mat_3[idx_pair, :] = nxt_row3
    
    print("finished pair {0}/{1}".format(idx_pair+1, num_pairs))
    
    # Increment index of data dimensionality pairs.
    idx_pair += 1


# Save matrices of results.
sio.savemat(RESULTS_SAVE_PATH + 'time1_' + rba_name + '.mat', {'time1' : time_mat_1})
sio.savemat(RESULTS_SAVE_PATH + 'time2_' + rba_name + '.mat', {'time2' : time_mat_2})
sio.savemat(RESULTS_SAVE_PATH + 'time3_' + rba_name + '.mat', {'time3' : time_mat_3})



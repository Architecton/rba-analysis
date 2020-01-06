import os
import numpy as np
import scipy.io as sio

for dirname in os.listdir('./eval-set'):
    print("dataset {0}:".format(dirname))
    print(sio.loadmat("./eval-set/" + dirname + "/data.mat")['data'].shape)


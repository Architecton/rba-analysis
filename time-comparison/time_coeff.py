import numpy as np
import scipy.io as sio
import sys
from sklearn.linear_model import LinearRegression

# Parse data from file path specified by script arguments.
time_data = sio.loadmat(sys.argv[1])[sys.argv[1].split(".")[0]]

# Initialize and fit linear regression model.
lr = LinearRegression()
x = np.arange(100, 10000, 100)  # Match with value in time_eval.py
lr.fit(x[np.newaxis].T, time_data)

# Print coefficient.
print("coeff = {0}".format(np.ravel(lr.coef_)[0]))


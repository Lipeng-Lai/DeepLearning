import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)

def compute_cost(X, y, w, b, lambda_ = 1):
    m, n = X.shape

    loss_sum = 0

    for i in range(m):

        z_wb = 0

        for j in range (n):
            z_wb_i_j = w[j] * X[i][j]

            z_wb += z_wb_i_j
            
        z_wb += b

        f_wb = sigmoid(z_wb)

        loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)

        loss_sum += loss
        
    total_cost = (1/m) * loss_sum

    return total_cost
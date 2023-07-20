import copy, math
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import  dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob
# plt.style.use('./deeplearning.mplstyle')

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# fig,ax = plt.subplots(1,1,figsize=(4,4))
# plot_data(X_train, y_train, ax)

# ax.axis([0, 4, 0, 3.5])
# ax.set_ylabel('$x_1$', fontsize=12)
# ax.set_xlabel('$x_0$', fontsize=12)
# plt.show()

def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]
        
        dj_db = dj_db + err_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw
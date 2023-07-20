import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

'''
使用Numpy构建一个小型神经网络。它将与你在Tensorflow中实现的“咖啡烘焙”网络相同。
'''

X, Y = load_coffee_data()
# print(X.shape, Y.shape)

# plt_roast(X, Y)

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")


g = sigmoid

'''定义my_dense()函数，该函数计算稠密层的激活'''
def my_dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)

    return (a_out)


def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return (a2)


W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )


def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m, 1))

    for i in range(m):
        p[i, 0] = my_sequential(X[i], W1, b1, W2, b2)
    
    return (p)


X_tst = np.array([
    [200, 13.9], # pos
    [200, 17]]) # neg
X_tstn = norm_l(X_tst) # 归一化

prediction = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)


yhat = np.zeros_like(prediction)
for i in range(len(prediction)):
    if prediction[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decision = \n{yhat}")

yhat = (prediction >= 0.5).astype(int)
print(f"decisions = \n{yhat}")


netf= lambda x : my_predict(norm_l(x),W1_tmp, b1_tmp, W2_tmp, b2_tmp)
plt_network(X,Y,netf)
plt.show()
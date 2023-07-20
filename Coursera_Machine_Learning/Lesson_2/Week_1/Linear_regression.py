import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.python.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
# plt.style.use('./deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

'''
用np线性规划模拟实现tensorflow的线性层原理
'''

X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

# fig, ax = plt.subplots(1,1)
# ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
# ax.legend( fontsize='xx-large')
# ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
# ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
# plt.show()

linear_layer = tf.keras.layers.Dense(units = 1, activation = 'linear')

# 注意，该层的输入必须是2-D的，所以我们将重塑它
a1 = linear_layer(X_train[0].reshape(1, 1))
print(a1)
'''
tf.Tensor([[-0.07]], shape=(1, 1), dtype=float32)
结果是一个形状为(1,1)或只有一个项的张量(数组的另一种名称)。
现在让我们看看权重和偏差。这些权重被随机初始化为小数字，偏差默认初始化为零。

'''

w, b = linear_layer.get_weights()
# print(f"w = {w}, b = {b}")

# 权重初始化为随机值，所以让我们将它们设置为一些已知值
set_w = np.array([[200]])
set_b = np.array([100])

linear_layer.set_weights([set_w, set_b])
# print(linear_layer.get_weights())


# compare
print(X_train[0].reshape(1, 1))
a1 = linear_layer(X_train[0].reshape(1, 1))
# print(a1)

alin = np.dot(set_w, X_train[0].reshape(1, 1)) + set_b
# print(alin)

prediction_tf = linear_layer(X_train)
prediction_np = np.dot(X_train, set_w) + set_b

# plt_linear(X_train, Y_train, prediction_tf, prediction_np)
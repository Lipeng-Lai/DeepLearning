import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
# %matplotlib widget
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)



def my_softmax(z):
    ez = np.exp(z)
    sm = ez / np.sum(ez)
    return (sm)


# plt_softmax(my_softmax)


# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)


# model = Sequential(
#     [ 
#         Dense(25, input_shape = (2, ), activation = 'relu'),
#         Dense(15, activation = 'relu'),
#         Dense(4, activation = 'softmax')    # < softmax activation here
#     ]
# )
# # model.compile(
# #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
# #     optimizer=tf.keras.optimizers.Adam(0.001),
# # )

# # loss = tf.cast(tf.keras.losses.sparse_categorical_crossentropy, tf)
# model.compile(optimizer= 'adam' , loss= tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# model.fit(
#     X_train,y_train,
#     epochs=10
# )
       

# p_nonpreferred = model.predict(X_train)
# print(p_nonpreferred [:2])
# print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))


model = Sequential(
    [
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')    # < softmax activation here
    ]
)
# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(0.001),
# )

# loss = tf.cast(tf.keras.losses.sparse_categorical_crossentropy, tf)
model.compile(optimizer= 'adam' , loss= tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(
    X_train,y_train,
    epochs=10
)

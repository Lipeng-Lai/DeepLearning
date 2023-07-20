import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
np.set_printoptions(precision=2)
from lab_utils_multiclass_TF import *
import logging
from tensorflow.python.keras.optimizers import adam_v2
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# make 4-class dataset for classification
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)


# plt_mc(X_train,y_train,classes, centers, std=std)


tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(2, activation='relu', name = 'L1'),
        Dense(4, activation= 'linear', name = 'L2')
    ]
)

# model.compile(optimizer= 'adam' , loss= tf.keras.losses.sparse_categorical_crossentropy(from_logits=True), metrics=['accuracy'])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=adam_v2.Adam(0.01)
)

model.fit(
    X_train, y_train,
    epochs = 200
)

# plt_cat_mc(X_train, y_train, model, classes)


l1 = model.get_layer("L1")
W1, b1 = l1.get_weights()

plt_layer_relu(X_train, y_train.reshape(-1, ), W1, b1, classes)


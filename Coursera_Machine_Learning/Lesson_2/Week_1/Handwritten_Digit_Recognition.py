import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *
# %matplotlib inline

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


X, y = load_data()

# print(str(X.shape))
# print(str(y.shape))

'''visualizing the Data'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

# m, n = X.shape

# fig, axes = plt.subplots(8,8, figsize=(8,8))
# fig.tight_layout(pad=0.1)

# for i,ax in enumerate(axes.flat):
#     # Select random indices
#     random_index = np.random.randint(m)
    
#     # Select rows corresponding to the random indices and
#     # reshape the image
#     X_random_reshaped = X[random_index].reshape((20,20)).T
    
#     # Display the image
#     ax.imshow(X_random_reshaped, cmap='gray')
    
#     # Display the label above the image
#     ax.set_title(y[random_index,0])
#     ax.set_axis_off()
# plt.show()


'''Tensorflow Model Implementation'''

model = Sequential(
    [               
        # tf.keras.Input(shape=(400,)),    #specify input size
        ### START CODE HERE ### 
        Dense(25, input_shape = (400, ), activation='sigmoid'), 
        Dense(15, activation='sigmoid'), 
        Dense(1,  activation='sigmoid')  
        ### END CODE HERE ### 
    ], name = "my_model" 
)

# model.summary()

[layer1, layer2, layer3] = model.layers

#### Examine Weights shapes
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
# print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
# print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
# print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")


# print(model.layers[2].weights)

# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(0.001),
# )
model.compile(optimizer= 'adam' , loss= tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

model.fit(
    X,y,
    epochs=20
)


prediction = model.predict(X[0].reshape(1,400))  # a zero
# print(f" predicting a zero: {prediction}")
prediction = model.predict(X[500].reshape(1,400))  # a one
# print(f" predicting a one:  {prediction}")


def my_dense(a_in, W, b, g):
    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    
    return (a_out)


x_tst = 0.1*np.arange(1,3,1).reshape(2,)  # (1 examples, 2 features)
W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
b_tst = 0.1*np.arange(1,4,1).reshape(3,)  # (3 features)
A_tst = my_dense(x_tst, W_tst, b_tst, sigmoid)
# print(A_tst)


def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x, W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)

    return a3


W1_tmp, b1_tmp = layer1.get_weights()
W2_tmp, b2_tmp = layer2.get_weights()
W3_tmp, b3_tmp = layer3.get_weights()


# make predictions
yhat = 0
prediction = my_sequential(X[0], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
if prediction >= 0.5:
    yhat = 1
else:
    yaht = 0

# print("yhat = ", yhat, " label = ", y[0, 0])


# warnings.simplefilter(action='ignore', category=FutureWarning)
# # You do not need to modify anything in this cell

# m, n = X.shape

# fig, axes = plt.subplots(8,8, figsize=(8,8))
# fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

# for i,ax in enumerate(axes.flat):
#     # Select random indices
#     random_index = np.random.randint(m)
    
#     # Select rows corresponding to the random indices and
#     # reshape the image
#     X_random_reshaped = X[random_index].reshape((20,20)).T
    
#     # Display the image
#     ax.imshow(X_random_reshaped, cmap='gray')

#     # Predict using the Neural Network implemented in Numpy
#     my_prediction = my_sequential(X[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
#     my_yhat = int(my_prediction >= 0.5)

#     # Predict using the Neural Network implemented in Tensorflow
#     tf_prediction = model.predict(X[random_index].reshape(1,400))
#     tf_yhat = int(tf_prediction >= 0.5)
    
#     # Display the label above the image
#     ax.set_title(f"{y[random_index,0]},{tf_yhat},{my_yhat}")
#     ax.set_axis_off() 
# fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
# plt.show()

# print(X.shape)
x = X[0].reshape(-1, 1)
# print(x.shape)
# print(W1.shape) # (400, 25)
z1 = np.matmul(x.T, W1) + b1 # (1, 400) matmul (400, 25)
a1 = sigmoid(z1)

# print(a1.shape) (1, 25)



def my_dense_v(A_in, W, b, g):

    Z = np.matmul(A_in, W) + b
    A_out = g(Z)

    return (A_out)


X_tst = 0.1*np.arange(1,9,1).reshape(4,2) # (4 examples, 2 features)
W_tst = 0.1*np.arange(1,7,1).reshape(2,3) # (2 input features, 3 output features)
b_tst = 0.1*np.arange(1,4,1).reshape(1,3) # (1, 3 features)
A_tst = my_dense_v(X_tst, W_tst, b_tst, sigmoid)
print(A_tst)
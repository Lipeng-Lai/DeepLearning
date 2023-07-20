import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import dlc, plot_data
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
from lab_utils_common import plot_data, sigmoid, draw_vthresh
# plt.style.use('./deeplearning.mplstyle')


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])
def test1():


    pos = y_train == 1
    neg = y_train == 0

    fig,ax = plt.subplots(1,2,figsize=(8,3))
    #plot 1, single variable
    ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
                edgecolors=dlc["dlblue"],lw=3)

    ax[0].set_ylim(-0.08,1.1)
    ax[0].set_ylabel('y', fontsize=12)
    ax[0].set_xlabel('x', fontsize=12)
    ax[0].set_title('one variable plot')
    ax[0].legend()

    #plot 2, two variables
    plot_data(X_train2, y_train2, ax[1])
    ax[1].axis([0, 4, 0, 4])
    ax[1].set_ylabel('$x_1$', fontsize=12)
    ax[1].set_xlabel('$x_0$', fontsize=12)
    ax[1].set_title('two variable plot')
    ax[1].legend()
    plt.tight_layout()
    plt.show()


def test2():
    w_in = np.zeros(1)
    b_in = 0
    plt.close('all')
    addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)
    plt.show()


def test3():
    input_array = np.array([1,2,3])
    exp_array = np.exp(input_array)

    print("Input to exp:", input_array)
    print("Output of exp:", exp_array)

    # Input is a single number
    input_val = 1  
    exp_val = np.exp(input_val)

    print("Input to exp:", input_val)
    print("Output of exp:", exp_val)


def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    
    return g


def test4():
    z_tmp = np.arange(-10, 11)

    y = sigmoid(z_tmp)

    np.set_printoptions(precision=3)
    print("Input (z), Output (sigmoid(z))")
    print(np.c_[z_tmp, y])

    # Plot z vs sigmoid(z)
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    ax.plot(z_tmp, y, c="b")

    ax.set_title("Sigmoid function")
    ax.set_ylabel('sigmoid(z)')
    ax.set_xlabel('z')
    draw_vthresh(ax,0)
    plt.show()


def test5():
    x_train = np.array([0., 1, 2, 3, 4, 5])
    y_train = np.array([0,  0, 0, 1, 1, 1])

    w_in = np.zeros((1))
    b_in = 0

    plt.close('all') 
    addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
    plt.show()


def test6():
    X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    plot_data(X, y, ax)

    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel('$x_1$')
    ax.set_xlabel('$x_0$')
    plt.show()


def test7():
    X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 
    # Choose values between 0 and 6
    x0 = np.arange(0,6)

    x1 = 3 - x0
    fig,ax = plt.subplots(1,1,figsize=(5,4))
    # Plot the decision boundary
    ax.plot(x0,x1, c="b")
    ax.axis([0, 4, 0, 3.5])

    # Fill the region below the line
    ax.fill_between(x0,x1, alpha=0.2)

    # Plot the original data
    plot_data(X,y,ax)
    ax.set_ylabel(r'$x_1$')
    ax.set_xlabel(r'$x_0$')
    plt.show()


test7()
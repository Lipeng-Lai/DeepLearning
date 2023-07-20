import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)

# def test1():
#     a = np.zeros(4)
#     b = np.random.random_sample(4)

#     print(a)

#     print(b)


# def test2():
#     a = np.array([5, 4, 3, 2])
#     b = np.array([5. , 4, 3, 2])

#     print(a.dtype), print(b.dtype)


# def test3():
#     a = np.arange(10)
#     print(a)

#     print(a[1].shape)


# def test4():

#     a = np.array([1, 2, 3, 4])
#     print(a)

#     b = -a
#     print(b)

#     sum = np.sum(a)
#     print(sum)

#     average = np.mean(a)
#     print(average)

#     Pow = a**2
#     print(Pow)


# def test5():
#     a = np.array([1, 2, 3, 4])
#     b = np.array([-1, -2, -3, -4])
    
#     print(a + b)


# def test6():
    
#     a = np.array([1, 2, 3, 4])
#     b = 5 * a
#     print(a)
#     print(b)


# def my_dot(a, b):
#     x = 0
#     for i in range(a.shape[0]):
#         x = x + a[i] * b[i]
#     return x

# # a = np.array([1, 2, 3, 4])
# # b = np.array([-1, 4, 3, 2])
# # print(f"my_dot(a, b) =  {my_dot(a, b)}")

# def test7():
#     a = np.array([1, 2, 3, 4])
#     b = np.array([-1, 4, 3, 2])
#     c = np.dot(a, b)
#     print(c)

#     d = np.dot(a, b)
#     print(d)


# def test8():
#     a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
#     print(f"a.shape: {a.shape}, \na= {a}")



# X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
# y_train = np.array([460, 232, 178])


def test1():
    X_train, y_train = load_house_data()
    X_features = ['size(sqft)','bedrooms','floors','age']
    fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Price (1000's)")
    # plt.show()

    #set alpha to 9.9e-7
    # _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
    # plot_cost_i_w(X_train, y_train, hist)

    # _,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 9e-7)
    # plot_cost_i_w(X_train, y_train, hist)

    #set alpha to 1e-7
    _,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)
    plot_cost_i_w(X_train,y_train,hist)


test1()
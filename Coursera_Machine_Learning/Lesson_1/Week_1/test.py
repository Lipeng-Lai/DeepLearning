import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import *
from matplotlib import font_manager
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients


x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
def test1():
    m = x_train.shape[0]
    # m = len(x_train)

    i = 0

    x_i = x_train[i]
    y_i = y_train[i]

    plt.scatter(x_train, y_train, marker='x', c='r')

    plt.title("Housing Prices")

    plt.ylabel("Price (in 1000s of dollars)")

    plt.xlabel("Size (1000 sqft)")

    plt.show()


w, b = 100, 100
def test2():

    def compute_model_output(x, w, b):

        m = x.shape[0]
        f_wb = np.zeros(m)

        for i in range(m):
            f_wb[i] = w*x[i] + b
        
        return f_wb
    
    tmp_f_wb = compute_model_output(x_train, w, b,)

    # 绘制点经过的曲线
    plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

    # 绘制散点
    plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

    plt.title("Housing Prices")

    plt.ylabel('Price (in 1000s of dollars)')

    plt.xlabel('Size (1000 sqft)')

    plt.legend()

    plt.show()


def test3():
    w = 200
    b = 100
    x_i = 1.2

    cost_1200_sqft = w*x_i + b
    print(f"${cost_1200_sqft:.0f} thousand dollars")


def test4():

    def compute_cost(x, y, w, b):
        m = x.shape[0]

        cost_sum = 0
        for i in range(m):
            f_wb = w*x[i] + b
            cost = (f_wb - y[i]) ** 2
            cost_sum = cost_sum + cost
        
        total_cost = (1 / (2*m)) * cost_sum
    
        return total_cost
    return compute_cost(x_train, y_train, w, b)


def test5():
    plt.close('all')
    fig, ax, dyn_items = plt_stationary(x_train, y_train)
    updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
    plt.show()


def test6():

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    plt.title('测试数据')
    plt.plot(x_train, y_train,label='测试',color='blue',linestyle='--')
    myfont = font_manager.FontProperties(fname="C:\Windows\Fonts\simfang.ttf")
    plt.legend(prop=myfont,loc="upper left") #表示在图中增加图例
    plt.style.use('bmh')#将背景颜色改为。。
    plt.show()


def test7():
    soup_bowl()


def test8():

    def compute_cost(x, y, w, b):
   
        m = x.shape[0] 
        cost = 0
        
        for i in range(m):
            f_wb = w * x[i] + b
            cost = cost + (f_wb - y[i])**2
        total_cost = 1 / (2 * m) * cost

        return total_cost

    def compute_gradient(x, y, w, b): 
        """
        Computes the gradient for linear regression 
        Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters  
        Returns
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
        """
        # Number of training examples
        m = x.shape[0]    
        dj_dw = 0
        dj_db = 0
        
        for i in range(m):  
            f_wb = w * x[i] + b 
            dj_dw_i = (f_wb - y[i]) * x[i] 
            dj_db_i = f_wb - y[i] 
            dj_db += dj_db_i
            dj_dw += dj_dw_i 
        dj_dw = dj_dw / m 
        dj_db = dj_db / m 
            
        return dj_dw, dj_db

        plt_gradients(x_train, y_train, compute_cost, compute_gradient)
        plt.show()

    return compute_gradient(x_train, y_train, w, b)



def test9():
    # initialize parameters
    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2
    # run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                        iterations, compute_cost, test8)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


test9()
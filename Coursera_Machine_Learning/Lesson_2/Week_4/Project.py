import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *

# %matplotlib inline

'''在本练习中，您将从头开始实现一个决策树，并将其应用于对蘑菇是否可食用或有毒进行分类的任务'''

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])


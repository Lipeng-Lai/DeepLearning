# for building linear regression models
from sklearn.linear_model import LinearRegression, Ridge

# import lab utility functions in utils.py
import utils 

x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('./c2w3_lab2_data1.csv')

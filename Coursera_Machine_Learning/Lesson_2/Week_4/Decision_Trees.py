import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])


def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    

# print(entroy(0.5))


def split_indices(X, index_feature):

    left_indeces = []
    right_indeces = []

    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indeces.append(i)
        else:
            right_indeces.append(i)
    
    return left_indeces, right_indeces


# print(split_indices(X_train, 0))


def weighted_entropy(X,y,left_indices,right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    
    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy

left_indices, right_indices = split_indices(X_train, 0)
# print(weighted_entropy(X_train, y_train, left_indices, right_indices))


def information_gain(X, y, left_indices, right_indices):
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)

    return h_node - w_entropy


# print(information_gain(X_train, y_train, left_indices, right_indices))


# for i, feature_name in enumerate(['EarShape', 'Face Shape', 'Whiskers']):
#     left_indices, right_indices = split_indices(X_train, i)
#     i_gain = information_gain(X_train, y_train, left_indices, right_indices)
#     print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")


tree = []
build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=1, current_depth=0, tree = tree)
generate_tree_viz([0,1,2,3,4,5,6,7,8,9], y_train, tree)
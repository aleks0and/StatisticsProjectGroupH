import numpy as np
import pandas as pd
from math import sqrt

pd.set_option("max_columns",50)


#Assignment 1 Numpy ex1
def inner_product(x,y):
    if type(x) == np.ndarray and type(y) == np.ndarray and len(x) == len(y):
        return (x*y).sum()
    else:
        print("The type of x or y is not np.array or their length is different")
        return None


#Assign ment 1 Numpy ex2
def mean_absolute_error(x, y):
    if type(x) == np.ndarray and type(y) == np.ndarray and len(x) == len(y):
        return (abs(x - y).sum()) / len(x)
    else:
        print("The type of x or y is not np.array or their length is different")


#Assignment 1 Numpy ex3
def Lead(x, n):
    if type(x) == np.ndarray and type(n) == int and len(x) > n:
        return (np.append(x[n:], ["NaN"] * n))
    else:
        print("The type of x is not np.array or the type of n is not int or n is bigger than the x's length")
        return None


def Lag(x, n):
    if type(x) == np.ndarray and type(n) == int and len(x) > n:
        vector = ["Nan"]* n
        return (np.append(vector,x[:(len(x)-n)]))
    else:
        print("The type of x is not np.array or the type of n is not int")
        return None


#Assignment 1 Numpy ex4
def point_pairwise_distance(x,y):
    if (type(x) == np.ndarray and type(y) == np.ndarray):
        return sqrt(((x-y)**2).sum())


def pairwise_distance(X,y):
    if type(X) == np.ndarray and type(y) == np.ndarray:
        if X.shape[1] == y.shape[1]:
            d = np.zeros(shape=(1, X.shape[0]))
            for i in range(X.shape[0]):
                dist = point_pairwise_distance(X[i],y)
                d[0,i] = dist
            return d
            #np.apply_along_axis(point_pairwise_distance(,y),) ?? can we apply this function ??
        else:
            print("the dimentions of X and y do not match")
            return None
    else:
        print("The type of x or y is not np.array")
        return None



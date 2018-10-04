import numpy as np
from math import sqrt


def inner_product(x, y):
    #x and y must be arrays and their lenght have to be the same
    if type(x) == np.ndarray and type(y) == np.ndarray and len(x) == len(y):
        return (x*y).sum()
    else:
        #otherwise, tell the user there is a problem
        print("The type of x or y is not np.array or their length is different")
        return None


#Assignment 1 Numpy ex2: calculate the mean absolute error between x and y
def mean_absolute_error(x, y):
    #x and y must be arrays and their lenght have to be the same
    if type(x) == np.ndarray and type(y) == np.ndarray and len(x) == len(y):
        return (abs(x - y).sum()) / len(x)
    else:
        #otherwise, tell the user there is a problem
        print("The type of x or y is not np.array or their length is different")
        return None


#Assignment 1 Numpy ex3

#deleting the first n observation from array x and adding n "NaN" at the end of x
def lead(x, n): 
    #x has to be a numpy array and n has to be an integer 
    #n should be less than the lenght of x
    if type(x) == np.ndarray and type(n) == int and len(x) > n:
        return np.append(x[n:], ["NaN"] * n)
    else:
        #tells the user that there is an error in the input
        print("The type of x is not np.array or the type of n is not int or n is bigger than the x's length")
        return None

#deleting the last n observation from array x and adding n "NaN" at the beginning of x
def lag(x, n): 
    #x has to be a numpy array and n has to be an integer 
    #n should be less than the lenght of x
    if type(x) == np.ndarray and type(n) == int and len(x) > n:
        vector = ["NaN"] * n
        return np.append(vector, x[:(len(x)-n)])
    else:
        #tells the user that there is an error in the input
        print("The type of x is not np.array or the type of n is not int or n is greater than the lenght of x")
        return None


#Assignment 1 Numpy ex4: calculate the pairwise distance between points in X and a vector y
        
#distance formula
def point_pairwise_distance(x, y):
    #x and y are vectors
    if type(x) == np.ndarray and type(y) == np.ndarray:
        return sqrt(((x-y)**2).sum())
    else:
        #tells user there is an error in the input
        print("The type of x or y is not np.array")
        return None


def pairwise_distance(x, y):
    #checks types of inputs: vectors
    if type(x) == np.ndarray and type(y) == np.ndarray and x.shape[1] == y.shape[1]:
            d = np.zeros(shape=(1, x.shape[0]))
            for i in range(x.shape[0]):
                dist = point_pairwise_distance(x[i], y)
                d[0, i] = dist
            return d
    #tells the user there is an error in the input
    else:
        print("The type of x or y is not np.array or the dimentions of X and y do not match")
        return None



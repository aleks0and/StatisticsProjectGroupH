import numpy as np
from numpy_excersises import inner_product, mean_absolute_error, lead, lag, point_pairwise_distance, pairwise_distance

#Assignment 1 Numpy ex1 tests:
def inner_product_test():
    ex1 = inner_product(np.r_[1, 2], np.r_[3, 4])
    assert(ex1 == 11)
    # wrong length of vectors
    ex1 = inner_product(np.r_[1], np.r_[3, 4])
    assert(ex1 is None)
    # wrong data type
    ex1 = inner_product("A string", np.r_[3, 4])
    assert(ex1 is None)
    # column Wrong dimentions of the matrix-> doesnt have to be covered
    ex1 = inner_product(np.c_[1, 2], np.r_[3, 4])
    assert(ex1 is None)
    a1 = np.r_[1, 2, 3, 4]
    a2= np.r_[1, 2, 3, 4]
    ex1 = inner_product(a1, a2)
    assert(ex1 == 30)
    ex1 = inner_product(a1.reshape(-1, 1), a2.reshape(-1, 1))
    assert(ex1 == 30)
    return  None

#Assignment 1 Numpy ex2 tests:
def mean_absolute_error_test():
    vector1 = np.r_[1, 2, 5, 8, 7]
    vector2 = np.r_[1, 2, 5, 8, 7]
    # two rows
    ex2 = mean_absolute_error(vector1, vector2)
    print(ex2)
    # two different vectors
    vector3 = np.r_[1, 2, 3, 4, 7]
    ex2 = mean_absolute_error(vector1, vector3)
    print(ex2)
    #two columns
    ex2 = mean_absolute_error(vector1.reshape(-1, 1), vector2.reshape(-1, 1))
    print (ex2)
    ex2 = mean_absolute_error(vector1.reshape(-1, 1), vector3.reshape(-1, 1))
    print(ex2)
    #wrong data type
    ex2 = mean_absolute_error("a string", vector2.reshape(-1, 1))
    print(ex2)
    # wrong length of one vector
    vector4 = np.r_[1,2,3]
    ex2 = mean_absolute_error(vector1, vector4)
    print(ex2)

#Assignment 1 Numpy ex3 tests:
def lead_and_lag_test():
    vector1 = np.r_[1, 2, 3, 4, 5]
    #deleting first three observations
    ex3 = lead(vector1, 3)
    print(ex3)
    # deleting first 0 observations
    ex3 = lead(vector1, 0)
    print(ex3)
    #checking if Lead works with wrong data types
    ex3 = lead("string", 3)
    print(ex3)
    # checking if Lead works with wrong data types2
    ex3 = lead(vector1, "string")
    print(ex3)
    # checking if Lead works with n > x.length
    ex3 = lead(vector1, 7)
    print(ex3)
    # deleting last 3 observations
    ex3 = lag(vector1, 3)
    print(ex3)
    # deleting last 0 observations
    ex3 = lag(vector1, 0)
    print(ex3)
    # checking if Lag works with wrong data types
    ex3 = lag("string", 3)
    print(ex3)
    # checking if Lag works with wrong data types2
    ex3 = lag(vector1, "string")
    print(ex3)
    # checking if Lag works with n > x.length
    ex3 = lag(vector1, 7)
    print(ex3)


#Assignment 1 Numpy ex4 tests:
def pairwise_distance_test():
    x = np.array([
    [1,  4,  0,  0],
    [1,  0,  0,  2],
    [0, 0, 1, 0]
    ])
    y = np.array([
    [0, 0, 0, 0]
    ])
    z = np.array([
    [1, 4, 0, 0]
    ])
    w = np.array([
    [0, 1, 2]
    ])
    # calculating a pairwise distance between two points
    ex4 = point_pairwise_distance(y, z)
    print(ex4)
    # calculating the pairwise distance between multiple points and point y
    ex4 = pairwise_distance(x, y)
    print(ex4)
    # testing if the function works for X and y dimentions not matching
    ex4 = pairwise_distance(x, w)
    print(ex4)
    # testing if the function works with wrong data type
    ex4 = pairwise_distance(x, "a string")
    print(ex4)
    # testing if the function works with wrong data type
    ex4 = pairwise_distance("a string", w)
    print(ex4)

    return None

from numpy_excersises_tests import inner_product_test, mean_absolute_error_test,lead_and_lag_test,pairwise_distance_test
from pandas_excersises import pandas_exercise1, pandas_exercise2, pandas_exercise3


def main():

    print("Numpy section of the assignment")
    print("inner product task\n")
    inner_product_test()
    print("mean absolute error task\n")
    mean_absolute_error_test()
    print("lag and lead task\n")
    lead_and_lag_test()
    print("pairwise distance task\n")
    pairwise_distance_test()

    print("\n")
    print("Pandas section of the assignment")
    print("Pandas excersise 1 using weather data\n")
    pandas_exercise1()
    print("Pandas excersise 1 using flights data\n")
    pandas_exercise2()
    print("Pandas excersise 1 using birthday data\n")
    pandas_exercise3()
    return 0

main()
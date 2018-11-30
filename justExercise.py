import numpy as np
from random import randint


def define_polynomial(a, b, c):
    def ploy(x):
        result = a*x**2 + b*x + c
        return result

    return ploy


p1 = define_polynomial(2, 3, 1)


def our_decorator(func):
    def function_wrapper(x):
        print("Before calling " + func.__name__)
        func(x)
        print("After calling " + func.__name__)
    return function_wrapper


@our_decorator
def foo(x):
    print("Hi, foo has been called with " + str(x))


# foo("Hi")

aa = [1, 2, 3, 5]
bb = [4, 5, 6, 7]

aa_s = set(aa)
bb_s = set(bb)

print(aa_s.intersection(bb_s))


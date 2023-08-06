# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 07:58:57 2022

@author: paesc


"""

# import pylab
import pylab as py

def function_1(x):
    return x**x - 100

def function_2(a, b, c, x):
    return a * x ** 2 - b * x + c

# tuple with initial guess (lower, upper)
# error tuple (absolute, relative)
    
def root_bisec(init, error, func):
    a, b = init
    c = 0.5 * (a + b)
    while func(c) != 0:

        c = 0.5 * (a + b)
        if func(c) > 0:
            b = c
        else:
            a = c
        if (abs(a - b) < error[0] and abs(a- b) / abs(c) < error[1]):
            return c

# def analytical(a, b, c, func):
#     x1 

 
root1 = root_bisec((-100,100),(0.1,0.1), function_1)
print("bisected root of function: ", root1)

root2 = root_bisec((-100,100),(0.1,0.1), function_2)
print("biscected root of function 2: ", root2)
print("analytical root of function 2")
            
            

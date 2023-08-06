# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:44:35 2022

@author: paesc
"""


# -*- coding: utf-8 -*-

# update on remark: included cases where f(a) > 0 or f(b) < 0

def bisection(func, a, b, accepted_error):
    """
    This function solves the unknown root of a funtion given the funtion, the root boundaries
    and the accepted error.

    Parameters
    ----------
    func : User function, entered as a string.
    a : Initial lower root boundary
    b : Initial upper root boundary
    accepted_error : Accepted absolute delta of boundaries

    Returns
    -------
    f : Lower und Upper root boundary within the accepted error and the absolute error
    at the final iteration

    """
    f = lambda x: eval(func)

    abs_error = abs(b - a)

    while abs_error > accepted_error:
        c = (b + a) / 2

        if f(b) * f(a) >= 0:
            print("Choose a and b such that f(b) * f(a) is lower than zero")
            break

        elif f(a) * f(c) < 0:
            a = c
            abs_error = (b - a)

        elif f(b) * f(c) < 0:
            b = c
            abs_error = (b - a)

        else:
            print("Sorry, something did not work. Please check if function is continuous!")
            break

    print("The absolute error is", abs_error)
    print("The lower boundary is", a, "and the upper one is", b)


# check whether function works with problem with known root

# bisection("x ** 2 - 100", 15, 5, 0.1)

# requested function to solve with bisection method provided by assesment

bisection("x ** x - 100", -1, 4, 0.00001)

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 07:34:07 2022

@author: paesc
"""


# import math
from math import sqrt


def recPhiStupid(n):
    """
    Returns phi**(n) using phi(n) = phi(n-2) - phi(n-1)
    using self-recursion.
    """
    if n == 0:
        return 1
    elif n == 1:
        return 0.5 * (sqrt(5) - 1)
    else:
        return recPhiStupid(n-2) - recPhiStupid(n-1)


def recPhiLoop(n):
    """
    Returns phi**(n) using phi(n) = phi(n-2) - phi(n-1)
    using a loop.
    """
    if n == 0:
        return 1
    elif n == 1:
        return 0.5 * (sqrt(5) - 1)
    else:
        phi2 = 1
        phi1 = 0.5 * (sqrt(5) - 1)
        for i in range(n-1):
            phiN = phi2 - phi1
            phi2 = phi1
            phi1 = phiN
        return phiN


def prodPhi(n):
    """Returns phi**(n)"""
    phiN = 1
    phi = 0.5 * (sqrt(5) - 1)
    for n in range(n):
        phiN *= phi
    return phiN


print("recPhiLoop(0): %f" % recPhiLoop(0))
print("recPhiLoop(1): %f" % recPhiLoop(1))
print("prodPhi(0): %f" % prodPhi(0))
print("prodPhi(1): %f" % prodPhi(1))

n = 10
print("\nrecPhiLoop(%i): %g" % (n, recPhiLoop(n)))
print("prodPhi(%i): %g" % (n, prodPhi(n)))

n = 20
print("\nrecPhiLoop(%i): %g" % (n, recPhiLoop(n)))
print("prodPhi(%i): %g" % (n, prodPhi(n)))

n = 30
print("\nrecPhiLoop(%i): %g" % (n, recPhiLoop(n)))
print("prodPhi(%i): %g" % (n, prodPhi(n)))

n = 50
print("\nrecPhiLoop(%i): %g" % (n, recPhiLoop(n)))
print("prodPhi(%i): %g" % (n, prodPhi(n)))

n = 60
print("\nrecPhiLoop(%i): %g" % (n, recPhiLoop(n)))
print("prodPhi(%i): %g" % (n, prodPhi(n)))

n = 80
print("\nrecPhiLoop(%i): %g" % (n, recPhiLoop(n)))
print("prodPhi(%i): %g" % (n, prodPhi(n)))

n = 100
print("\nrecPhiLoop(%i): %g" % (n, recPhiLoop(n)))
print("prodPhi(%i): %g" % (n, prodPhi(n)))
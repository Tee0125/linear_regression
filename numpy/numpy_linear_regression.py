#!/usr/bin/env python

import numpy

x = numpy.array([1.1, 2.1, 4.3, -1.2, -2.4, -3.5])
y = numpy.array([0.7, 1.0, 3.2, -1.1, -2.1, -3.4])

o = numpy.ones(x.shape)

X  = numpy.array([x, o])
XT = numpy.transpose(X)

Y  = numpy.array([y])

# A * [ X, 1 ] = Y

# => A * [ X, 1 ] * [ X, 1 ]T = Y * [ X , 1 ]T
YXT = numpy.matmul(Y, XT)

# => A = Y * [ X , 1 ]T * ([ X, 1 ] & [ X, 1 ]T)inv
XXT     = numpy.matmul(X, XT)
XXT_INV = numpy.linalg.inv(XXT)

A = numpy.matmul(YXT, XXT_INV)

print(A)

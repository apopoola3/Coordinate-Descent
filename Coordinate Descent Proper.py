#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:50:14 2021

@author: anjolaoluwapopoola
"""

import numpy as np


# =============================================================================
# Using random data
# =============================================================================
A = np.array([[1,2],[3,4]])
A.shape
A_num_rows = A.shape[0] #number of rows in the data
A_num_features = A.shape[1]  #number of columns in the data
y = np.array([[-2],[3]])
lambdaa = 0.5
n = np.arange(0,A_num_features) #This is the amount of coeffs im expecting- in this case, i have two features, hence, x1 and x2 

# =============================================================================
# Initializing
# =============================================================================
x = np.zeros(2)
k = 0
e = 1
L = [5]
e_k = [  1]

# =============================================================================
# Writing Soft threshold function
# =============================================================================

def softhres(x,gamma):
    
    if x > gamma:
        s = x - gamma
        
    elif x < -gamma:
        s = x + gamma
    
    else:
        s = 0
     
    return s


# =============================================================================
# Writing coordinate descent code for re-iteration
# =============================================================================



while e > 0.0001:
    k = k + 1
    for i in n:
       Ai = A[:,i].reshape((A_num_rows,1))
       A_i = A[:,n!=i].reshape((A_num_rows,A_num_features-1))
       ata = np.matmul(Ai.transpose(),Ai)
       gamma = lambdaa / ata
       az1 = y - (A_i*x[n!=i])
       az3 = np.matmul(Ai.transpose(),az1)
       z = np.divide(az3,ata)
       x[i] = softhres(z,gamma)
        
    l_k = (0.5 * (np.linalg.norm(y.transpose() - (np.matmul(A,x.transpose()))))**2) +(lambdaa * np.linalg.norm(x,1))
    L.append(l_k)
    e = L[k-1] - L[k]
    
print(x)


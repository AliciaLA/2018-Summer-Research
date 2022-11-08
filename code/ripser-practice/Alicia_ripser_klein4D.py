#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:52:20 2018

@author: alicia
"""
import numpy as np
import scipy.spatial
import scipy as sp

'''
    Creating some toy data in R^4 to make a klein bottle in 4D
'''

#start of by defining the theta and phi as parameters
parameters = [[np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)] for i in range(1,500)]

#the klein equations using the respective parameters
#par[0]=theta
#par[1]=phi

'''
    I retrieved these equations from: 
        https://www.mathcurve.com/surfaces.gb/klein/klein.shtml
'''

a, b = 1, 2
klein = [[(a+b*np.cos(par[1]))*np.cos(par[0]), (a+b*np.cos(par[1]))*np.sin(par[0]),
          b*np.sin(par[1])*np.cos(par[0]/2), b*np.sin(par[1])*np.sin(par[0]/2)] for par in parameters]

#now define the data array
data = np.array(klein, dtype=float)

#creating our distance matrix
dist_matrix = sp.spatial.distance.squareform(sp.spatial.distance.pdist(data, metric='euclidean'))

#creating our csv file to later use in live ripser dot org
np.savetxt('Alica_ripser_klein4D.csv', dist_matrix, delimiter=',')
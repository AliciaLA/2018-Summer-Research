#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:22:51 2018

@author: alicia
"""
import numpy as np
import scipy.spatial
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

'''
    Creating some toy data in R^3 to make a klein bottle using 
    figure-8 parametric equations
'''

#start of by defining the theta and phi as parameters
parameters = [[np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)] for i in range(1,2000)]

#the klein equations using the respective parameters
#par[0]=theta
#par[1]=phi
a = 3
klein = [[(a+np.cos((1/2)*par[0])*np.sin(par[1])-np.sin((1/2)*par[0])*np.sin(2*par[1]))*np.cos(par[0]),
          (a+np.cos((1/2)*par[0])*np.sin(par[1])-np.sin((1/2)*par[0])*np.sin(2*par[1]))*np.sin(par[0]),
          np.sin((1/2)*par[0])*np.sin(par[1])+np.cos((1/2)*par[0])*np.sin(2*par[1])] for par in parameters]

#now define the data array
data = np.array(klein, dtype=float)

#might as well plot and see the intersections
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], s=10) 
ax.view_init(azim = 100, elev = 20)
ax.set_xlim3d(-5,5)
ax.set_ylim3d(-5,5)
ax.set_zlim3d(-5,5)

#creating our distance matrix
dist_matrix = sp.spatial.distance.squareform(sp.spatial.distance.pdist(data, metric='euclidean'))

#creating our csv file to later use in live ripser dot org
np.savetxt('Alica_ripser_klein_figure8.csv', dist_matrix, delimiter=',')
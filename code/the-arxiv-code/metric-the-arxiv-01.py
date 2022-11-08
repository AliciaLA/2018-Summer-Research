#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:18:13 2018

@author: alicia
"""
import pandas as pd
import numpy as np
import scipy.spatial


processed_df = pd.read_csv('arxiv_data_TDA.csv', index_col = False)

# Proceed with 3D array 
numerical_df = processed_df.copy()
#numerical_df['author_count'] = pd.Series(map(lambda x: len(x), numerical_df['authors']))
numerical_df = numerical_df[['weights','pages', 'references']].values

# Define function to filter out nonzero values
def allnonzero(lst):
    return reduce(lambda x, y: (x != 0) and (y != 0), lst)

#numerical_df = filter(lambda x: allnonzero(x), numerical_df)

#numerical_array = np.asarray(numerical_df)

# This is how we define our metric:
def metric(x, y):
    return np.sum(np.abs((1.0/(x+1.0))-(1.0/(y+1)))

# Create distance metric to later use in Dionysus: 
# dist = scipy.spatial.distance.pdist(numerical_df, metric)

'''
This metric is proved analytically.
'''
# End  

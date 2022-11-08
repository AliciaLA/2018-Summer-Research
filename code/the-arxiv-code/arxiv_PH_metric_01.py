#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:22:51 2018

@author: alicia
"""
import pandas as pd
import numpy as np
import dionysus as dio
import scipy.spatial

processed_df = pd.read_csv('arxiv_data.csv', index_col = False)

numerical_df = processed_df.copy()
#numerical_df['author_count'] = pd.Series(map(lambda x: len(x), numerical_df['authors']))
#numerical_df = numerical_df[['author_count','pages', 'references']].values 
numerical_df = numerical_df[['weights','pages', 'references']].values

def metric(x, y):
    return np.sum(np.abs((1.0/(x+1.0))-(1.0/(y+1.0))))
#np.sum(np.abs(x-y)) with weights

dist = scipy.spatial.distance.pdist(numerical_df, metric)

# Apply Dionysus and create filtration 
sample_filtration = dio.fill_rips(dist, 2, 300)

# Create persistence 
sample_persistence = dio.homology_persistence(sample_filtration)

# Create diagram
diagram_info = dio.init_diagrams(sample_persistence, sample_filtration)

# Plot persistence diagrams and bars		     
dio.plot.plot_diagram(diagram_info[0], show=True)
dio.plot.plot_bars(diagram_info[0], order='death', show=True)
dio.plot.plot_diagram(diagram_info[1], show=True)
dio.plot.plot_bars(diagram_info[1], order='death', show=True)

# End

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:27:09 2018

@author: alicia
"""

import pandas as pd
import numpy as np
import kmapper as km
from sklearn import ensemble 
import sklearn as sk
import time
import scipy 
import re

# READ CSV FILE AND COPY DATAFRAMES
processed_df = pd.read_csv('data.csv', nrows = 100, index_col = False)

numerical_df = processed_df.copy()
category_df = processed_df.copy()

# PROCESS DATAFRAMES
cat_df = category_df[['authors','summary','published_parsed']] # CATEGORICAL
numerical_df = numerical_df[['weights', 'pagenum', 'refnum']] # NUMERICAL

# CREATE EMPTY LISTS AND APPEND
list_authors=[]
for item in category_df['authors']:
        list_authors.append(set(filter(lambda a: a != '[', re.findall('\'(.*?)\'', item))))
list_summary=[]
for item in category_df['summary']:
    delimeters = ' and ', ' or ', ' ', ',', ';', '&', '(?)', '(', ')', '/', '|', '.', '\n'
    regexPattern = '|'.join(map(re.escape, delimeters))
    list_summary.append(set(filter(lambda a: a != '', re.split(regexPattern, item))))

# CREATE DATAFRAME FOR AUTHOR AND SUMMARY STRINGS
cat_string_df = pd.DataFrame()
cat_string_df['authors'] = list_authors
cat_string_df['summary'] = list_summary

# CREATE NEW COLUMNS IN NUMERICAL DATAFRAME
len_authors=[]
for x in cat_string_df['authors']:
    len_authors.append(len(x))
len_summary=[]
for y in cat_string_df['summary']:
    len_summary.append(len(y))

numerical_df['len_authors'] = np.array(len_authors)
numerical_df['len_summary'] = np.array(len_summary)

# DEFINE FUNCTION TO RETRIEVE SUM OF SECONDS OF ALL TIME STRUCTURE COMPONENTS 
def str_to_secs(date_str):
	head = date_str.index('(') + 1
	tail = date_str.index(')')
	return time.mktime(tuple(map(lambda x: int(x[x.index('=') + 1:]),
	                             date_str[head:tail].split(','))))

# USE DEFINITION TO RETRIEVE TIME VALUE OF EACH PAPER	
list_int = map(lambda x: str_to_secs(x), cat_df['published_parsed'])
avg = reduce(lambda x,y: x+y, list_int)/len(list_int)
numerical_df['time'] = pd.Series(map(lambda x: x/avg, list_int))
             
# CREATE A COLOR FUNCTION
# FIRST PROCESS ACADEMIC CATEGORIES 

categories = processed_df.copy()['category']

# REDUCE THE CATEGORIES DOWN TO TYPE OF SUBJECT
# THIS IS TO REDUCE THE NUMBER OF COLORS

reduced_categories = []

for x in categories: 
    if 'physics' in x or 'hep' in x or x == 'math-ph' or x == 'quant-ph':
        reduced_categories.append('physics')
    elif 'math' in x: 
        reduced_categories.append('mathematics')
    elif 'cond' in x: 
        reduced_categories.append('condensed matter')
    elif 'nlin' in x: 
        reduced_categories.append('nonlinear sciences')
    elif 'cs' in x: 
        reduced_categories.append('computer science')
    elif x == 'astro-ph' or x == 'gr-qc': 
        reduced_categories.append('Universe')
    elif 'nucl' in x: 
        reduced_categories.append('Nuclear')
    else: 
        reduced_categories.append('NONE FOUND')
        print(x)

# CONVERTING THE CATEGORIES TO NUMERICAL VALUES IN AN ARRAY

red_cat_nums = pd.DataFrame(data = reduced_categories, columns = ['cat_nums'])['cat_nums'].astype('category').cat.codes
my_colors = red_cat_nums.values

# DEFINE METRICS 

def mymetric(x, y):
    return np.sum(np.abs((1.0/(x+1.0))-(1.0/(y+1.0))))

def similarity_metric(a, b):
	score = 0
	try:
		for i in [0, 1]:
			intersection = a[i] & b[i]
			union = a[i] | b[i]
			score = score + (1 - (len(intersection) / len(union)))
	except:
		print("error with sim. met. fn. with pairs " + str(a) + str(b))
	return score

# CREATE DISTANCE MATRIX

# NUMERICAL DISTANCE MATRIX
num_arr = np.array(numerical_df)
dist_num = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(num_arr,
                    metric = mymetric))

# MAPPER ONLY ACCEPTS NUMERICAL VALUES
X = num_arr

# INITIALIZE KEPLER MAPPER
mapper = km.KeplerMapper(verbose=1)

# ISOLATION FOREST LENS
model = ensemble.IsolationForest()
model.fit(X)

# MULTIDIMENSIONAL SCALING LENS
mds = sk.manifold.MDS(dissimilarity = 'precomputed', random_state = 100)

# DEFINE LENSES
lens1 = model.decision_function(X).reshape(X.shape[0], 1) 
lens2 = mds.fit_transform(dist_num) 
lens3 = mapper.fit_transform(X, projection=[1]) 

# CONCATENATE LENSES
lens = np.c_[lens1, lens2, lens3]

# INITIALIZE DBSCAN AS CLUSTERING ALGORITHM
# AND CHOOSE METRIC
clusterer = sk.cluster.DBSCAN(eps = 0.5, min_samples = 2, mymetric)

# CREATE SIMPLICIAL COMPLEX 
simplicial_complex = mapper.map(lens, X, clusterer, 
                                cover = km.Cover(n_cubes=15, perc_overlap=0.9))

# CREATE HTML TO SEE MAPPER OUTPUT, AND APPLY COLOR FUNCTION
html = mapper.visualize(simplicial_complex, 
                        path_html="arxiv-final.html", 
                        title='My Metric', color_function=my_colors)

# End
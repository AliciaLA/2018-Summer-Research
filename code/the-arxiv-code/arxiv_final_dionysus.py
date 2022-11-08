#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:10:04 2018

@author: alicia
"""

import pandas as pd
import numpy as np
import time
import scipy.spatial
import dionysus as dio
import re

# READ CSV FILE AND COPY DATAFRAMES
processed_df = pd.read_csv('data.csv', index_col = False)

# CHOOSE RANDOM SAMPLE
subset = processed_df.copy().sample(1000, random_state=199)

numerical_df = subset
category_df = subset

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
subset['authors'] = list_authors
subset['summary'] = list_summary

# CREATE NEW COLUMNS IN NUMERICAL DATAFRAME
len_authors=[]
for x in subset['authors']:
    len_authors.append(len(x))
len_summary=[]
for y in subset['summary']:
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

# DEFINE MYMETRIC FOR NUMERICAL VALUES
def mymetric(x, y):
    return np.sum(np.abs((1.0/(x+1.0))-(1.0/(y+1.0))))

num_arr = np.array(numerical_df)
dist_num = scipy.spatial.distance.pdist(num_arr,
                    metric = mymetric)

'''
# DEFINE SIMILARITY METRIC FOR STRINGS
keys = range(100)
authors_dict = dict(zip(keys, list_authors))
summaries_dict = dict(zip(keys, list_summary))

def similarity_metric(a, b):
    intersection_authors = authors_dict.get(a[0]).intersection(authors_dict.get(b[0]))
    union_authors = authors_dict.get(a[0]).union(authors_dict.get(b[0]))
    intersection_summaries = summaries_dict.get(a[1]).intersection(summaries_dict.get(b[1]))
    union_summaries = summaries_dict.get(a[1]).union(summaries_dict.get(b[1]))
    score1 = 1 - (float(len(intersection_authors))) / float(len(union_authors))
    score2 = 1 - (float(len(intersection_summaries))) / float(len(union_summaries))
    return score1 + score2

temp=pd.DataFrame()
temp['authors']=keys
temp['summaries']=keys

cat_arr = np.array(temp)
dist_cat = scipy.spatial.distance.pdist(cat_arr,
		           metric = similarity_metric)
'''
# APPLY DIONYSUS AND CREATE FILTRATION

sample_filtration = dio.fill_rips(dist_num, 2, 0.5)

# CREATE PERSISTENCE
sample_persistence = dio.homology_persistence(sample_filtration)

# CREATE DIAGRAM
diagram_info = dio.init_diagrams(sample_persistence, sample_filtration)

# PLOT PERSISTENT DIAGRAM AND BARCODES	
for i in range(len(diagram_info)):
		title = "My metric" + ", Dimension " + str(i)
		try:
			print("showing " + title)
			dio.plot.plot_diagram(diagram_info[i], show=True)
			dio.plot.plot_bars(diagram_info[i], order='death', show=True)
		except ValueError:
			print("No Diagram Available with metric: " +
"My metric" + ", and dimension: " + str(i))   

# HOW LONG DOES THIS TAKE?
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

# END
import pandas as pd
import numpy as np
import kmapper as km
from sklearn import *
import time

def main():
	# read in data file
	df = pd.read_csv('data.csv', index_col = False)
	
	# choose which fields to extract from the raw data set
	fields = ['authors', 'published_parsed', 'summary',
	          'weights', 'pagenum', 'refnum']
	          
	# convert to numpy array
	vals = np.asarray(df[fields].values)
	
	# string parsing to convert to numerical values
	for i in vals:
		if(isinstance(i[0], str)):
			i[0] = len(i[0].split(','))
		if(isinstance(i[1], str)):
			i[1] = str_to_secs(i[1])
		if(isinstance(i[2], str)):
			i[2] = len(i[2].split(' '))
	
	# MAPPER algorithm
	X = np.asarray(vals)
	
	mapper = km.KeplerMapper(verbose = 1)
	
	model = ensemble.IsolationForest()
	model.fit(X)
	
	#lens1 = model.decision_function(X).reshape(X.shape[0],1)
	lens2 = mapper.fit_transform(X, projection = "L2norm")
	#lens3 = mapper.fit_transform(X, projection=[0])
	
	lens = np.c_[lens2]
	simplicial_complex = mapper.map(lens, X, cover = km.Cover(n_cubes=15, perc_overlap = 0.5), clusterer = cluster.KMeans(n_clusters = 2))

	mapper.visualize(simplicial_complex,
	                 path_html = "mappersimplex_all.html",
	                 title = "arxiv data")

def str_to_secs(date_str):
	head = date_str.index('(') + 1
	tail = date_str.index(')')
	return time.mktime(tuple(map(lambda x: int(x[x.index('=') + 1:]),
	                             date_str[head:tail].split(','))))

main()

import pandas as pd
import numpy as np
import time, functools, math
import scipy.spatial
import dionysus as ds

def main():
	df = pd.read_csv('data.csv')
	
	fields = ['authors', 'published_parsed', 'summary',
	          'weights', 'pagenum', 'refnum']
	vals = np.asarray(df[fields].values)
	
	metrics = ['euclidean', 'cityblock', 'chebyshev', 'similarity']
	
	for i in metrics:
		diagrams(vals, i)
	
def diagrams(vals, metric_fn):
	dist_mat = []
	if(metric_fn == 'similarity'):
		dist_mat = scipy.spatial.distance.pdist(vals,
		           metric = similarity_metric)
	else:
		vals_copy = vals.copy()
		for i in vals_copy:
			if(isinstance(i[0], str)):
				i[0] = len(i[0].split(','))
			if(isinstance(i[1], str)):
				i[1] = str_to_secs(i[1])
			if(isinstance(i[2], str)):
				i[2] = len(i[2].split(' '))
		dist_mat = scipy.spatial.distance.pdist(vals_copy,
		           metric = metric_fn)
	
	filt = ds.fill_rips(dist_mat, 2, 30)
	pers = ds.homology_persistence(filt)
	diagram_info = ds.init_diagrams(pers, filt)
	for i in range(len(diagram_info)):
		title = metric_fn + ", Dimension " + str(i)
		try:
			print("showing " + title)
			ds.plot.plot_diagram(diagram_info[i], show=True)
			ds.plot.plot_bars(diagram_info[i], show=True)
		except ValueError:
			print("No Diagram Available with metric: " +
                  str(metric_fn) + ", and dimension: " + str(i))

def similarity_metric(a, b):
	score = 0
	try:
		for i in [0, 2]:
			intersection = list(set(a[i]) & set(b[i]))
			union = list(set(a[i]) | set(b[i]))
			score = score + (1 - (len(intersection) / len(union)))
	except:
		print("error with sim. met. fn. with pairs " + str(a) + str(b))
	return score
	
def str_to_secs(date_str):
	head = date_str.index('(') + 1
	tail = date_str.index(')')
	return time.mktime(tuple(map(lambda x: int(x[x.index('=') + 1:]),
	                             date_str[head:tail].split(','))))

main()

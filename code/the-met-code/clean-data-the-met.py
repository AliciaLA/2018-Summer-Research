""" Clean Data, Define Metric and Create Distance Matrix """
import pandas as pd
import numpy as np
import kmapper
import sklearn
from kmapper import KeplerMapper
from sklearn import ensemble
from sklearn.metrics import jaccard_similarity_score
import re
import scipy as sp

''' Keep the desirable features and Generate a random sample of 1000 artworks'''
df = pd.read_csv('MetObjects.csv', encoding="ISO-8859-1")
feature_names = ['Is Highlight', 'Is Public Domain', 'Object ID', 'Department', 'Object Name', 'Object Begin Date', 'Object End Date', 'Medium', 'Dimensions', 'Credit Line', 'Classification']
X = df[feature_names].dropna(axis=0)
sub = X.sample(1000)

''' Focus on artworks with 2 measurements and replace their Dimensions with area '''
for index, row in sub.iterrows():
    l = re.findall('\((.*?) cm', row[8])
    if len(l) == 1:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", l[0])
        if len(nums) == 2:
            sub.loc[index, 'Dimensions'] = float(nums[0]) * float(nums[1])
        else:
            sub.drop(index, inplace=True)
    else:
        sub.drop(index, inplace=True)

''' Initialize dictionaries for 'Object Name', 'Medium', 'Credit Line', 'Classification' '''
mydict_name={}
mydict_medium={}
mydict_cl={}
mydict_class={}

''' Convert the type of these variables (plus 'Department') to 'category', so that we can easily assign them numerical values '''
sub['Department'] = sub['Department'].astype('category')
sub['Object Name'] = sub['Object Name'].astype('category')
sub['Medium'] = sub['Medium'].astype('category')
sub['Credit Line'] = sub['Credit Line'].astype('category')
sub['Classification'] = sub['Classification'].astype('category')

''' Create new columns in sub and set them to the numerical values of the corresponding columns '''
sub['depart_cat'] = sub['Department'].cat.codes
sub['name_cat'] = sub['Object Name'].cat.codes
sub['med_cat'] = sub['Medium'].cat.codes
sub['cl_cat'] = sub['Credit Line'].cat.codes
sub['class_cat'] = sub['Classification'].cat.codes

''' Record the correspondence between the numerical values and strings for each of the variables.
    We don't do so for 'Deparment' because of the way we give a subscore for it. '''
''' Still need to improve: delimiters '''
delimiters = ' and ', ' or ', ' ', ',', ';', '&', '(?)', '(', ')', '/', '|', '.'
regexPattern = '|'.join(map(re.escape, delimiters))
for idx, item in enumerate(sub['name_cat']):
    mydict_name[item]=list(filter(lambda a: a != '', re.split(regexPattern, sub['Object Name'][idx])))
for idx, item in enumerate(sub['med_cat']):
    mydict_medium[item]=list(filter(lambda a: a != '', re.split(regexPattern, sub['Medium'][idx])))
for idx, item in enumerate(sub['cl_cat']):
    delims = ', ', '; '
    rePattern = '|'.join(map(re.escape, delims))
    mydict_cl[item]=list(filter(lambda a: a != '', re.split(rePattern, sub['Credit Line'][idx])))
for idx, item in enumerate(sub['class_cat']):
    mydict_class[item]=list(filter(lambda a: a != '', re.split(regexPattern, sub['Classification'][idx])))
    '''For Classification, we need to further refine the lists of meaningful words, 
        since the words after '-' seem similar to an artwork's Object Name and we decided to discard the part after '-'''
    mydict_class[item]=list(map(lambda x: x.split('-')[0], mydict_class.get(item)))

''' Determine the maximum difference in areas '''
max_diff_area = 0
for x in sub['Dimensions']:
    for y in sub['Dimensions']:
        if abs(x-y) > max_diff_area:
            max_diff_area = abs(x-y)

''' Drop the columns with string values and keep only the numerical columns '''
sub = sub.drop(columns=['Department', 'Object Name', 'Medium', 'Credit Line', 'Classification'])
sub=np.array(sub)
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:21:58 2018

@author: Hongyuan
"""
def get_categories(sub): 
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
    for idx, item in enumerate(sub['name_cat']):
        mydict_name[item]=sub['Object Name'][idx]
    for idx, item in enumerate(sub['med_cat']):
        mydict_medium[item]=sub['Medium'][idx]
    for idx, item in enumerate(sub['cl_cat']):
        mydict_cl[item]=sub['Credit Line'][idx]
    for idx, item in enumerate(sub['class_cat']):
        mydict_class[item]=sub['Classification'][idx]
    
    ''' Determine the maximum difference in areas '''

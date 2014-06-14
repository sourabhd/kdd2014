# Adapted from YCSU : http://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/forums/t/8280/school-locations-and-poverty-levels-on-a-map

from mpl_toolkits.basemap import Basemap
import pandas as pd
import numpy as np
import pylab

def relabel(arr, order=None, target=None):
    if order:
        cat = order
    else:    
        cat = np.sort(list(set(arr)))
    if target:
        catDic = dict(zip(cat, target))
    else:
        catDic = dict(zip(cat, range(len(cat))))
    res = []
    for label in arr:
        res.append(catDic[label])
    return np.array(res)

m = Basemap(projection='mill', llcrnrlat=20, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-65)
projects_all = pd.read_csv(open('../../dataset/projects.csv','r'))
outcomes = pd.read_csv(open('../../dataset/outcomes.csv','r'))
projects = pd.merge(projects_all,outcomes,on='projectid')

lon = projects['school_longitude'].values
lat = projects['school_latitude'].values
x, y = m(lon, lat)

m.drawcoastlines()
m.drawcountries(linewidth=2.0, linestyle='--', color='r')
m.drawstates()
#m.scatter(x, y, c=relabel(projects['poverty_level'], order=['low poverty', 'moderate poverty', 'high poverty', 'highest poverty'], target=['b','g','y','r'] ), alpha=0.1)
#m.scatter(x, y, c=relabel(projects['is_exciting'], order=['t', 'f'], target=['b','g','y','r'] ), alpha=0.1)
m.scatter(x, y, c=relabel(projects[projects['is_exciting'] == 'f']['is_exciting'], order=['t', 'f'], target=['b','g','y','r'] ), alpha=0.1)

pylab.show()

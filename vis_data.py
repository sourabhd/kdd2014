from __future__ import print_function
import pandas as pd
import numpy as np
import pylab
import pdb
import sys
import matplotlib.pyplot as plt
import pandas.tools.rplot as rplot

def main():
    projects_all = pd.read_csv(open('../../dataset/projects.csv','r'))
    outcomes = pd.read_csv(open('../../dataset/outcomes.csv','r'))
    projects = pd.merge(projects_all,outcomes,on='projectid')
    projects1 = pd.DataFrame(projects.fillna(''),columns=['projectid','primary_focus_area','is_exciting'])
    projects1['cat_primary_focus_area'] = pd.factorize(projects1.primary_focus_area)[0]
    print(projects1)
    plt.figure()
    plot = rplot.RPlot(projects1, x='cat_primary_focus_area',)
    plot.add(rplot.TrellisGrid(['.', 'is_exciting']))
    plot.add(rplot.GeomHistogram())
    plot.render(plt.gcf())
    pylab.show()

if __name__ == '__main__':
    try:
        main()
    except:
        pdb.post_mortem(sys.exc_info()[2])


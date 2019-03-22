#!/usr/bin/env python
# coding: utf-8

# ### Identifying Clusters

## let's take a look at some different sets of data to practice identifying clusters.


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import helper_functions as h
import test_file as t
from IPython import display

#get_ipython().run_line_magic('matplotlib', 'inline')

# Make the images larger
plt.rcParams['figure.figsize'] = (16, 9)

print(h.plot_q1_data())
#1 clusters in this plot = 4

print(h.plot_q2_data())
#2 clusters in this plot = 2

print(h.plot_q3_data())
#3 clusters in this plot = 6

print(h.plot_q4_data())
#4 clusters in this plot = 7

'''This data(#4) is actually the same as the data used in #3.
 It shows how looking at data from a different angle can make us believe there
 are a different number of clusters in the data! '''
#!/usr/bin/env python
# coding: utf-8

# ### Changing K
# 
# lets do some practice using different values of k in the k-means algorithm, 
# and see how this changes the clusters that are observed in the data. 
# lets see what the best value for k might be for a dataset.
# 
# To get started, let's read in our necessary libraries.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import helpers2 as h
import tests as t
from IPython import display

# Make the images larger
plt.rcParams['figure.figsize'] = (16, 9)

# To get started, there is a function called **simulate_data** within the **helpers2** module. 
# Read the documentation on the function in helper2.py file.  Then use the function to 
# simulate a dataset with 200 data points (rows), 5 features (columns), and 4 centers

data = h.simulate_data(n = 200, features = 5, centroids = 4)

# Let's try a few different values for **k** and fit them to our data using **KMeans**.
# 
# To use KMeans, we need to follow three steps:
# 
# **I.** Instantiate the model.
# **II.** Fit model to the data.
# **III.** Predict the labels for the data.

# Try instantiating a model with 4 centers
kmeans_4 = KMeans(4)  #instantiate your model

# Then fit the model to data using the fit method
model_4 = kmeans_4.fit(data) #fit the model to data using kmeans_4

# Finally predict the labels on the same data to show the category that point belongs to
labels_4 =  model_4.predict(data) #predict labels using model_4 on dataset

# this should provide a plot of data colored by center
(h.plot_data(data, labels_4))


# Now try again, but this time fit kmeans using 2 clusters instead of 4 to data.

kmeans_2 = KMeans(2)
model_2 = kmeans_2.fit(data)
labels_2 = model_2.predict(data)
(h.plot_data(data, labels_2))


# Now try one more time, but with the number of clusters in kmeans to 7.

kmeans_7 = KMeans(7)
model_7 = kmeans_7.fit(data)
labels_7 = model_7.predict(data)
(h.plot_data(data, labels_7))

'''
usually, we get some indication of how well our model is doing, but it isn't totally apparent. 
Each time additional centers are considered, the distances between the points and the center will decrease.
However, at some point, that decrease is not substantial enough to suggest the need for an additional cluster.'''  

'''Using a scree plot is a common method for understanding if an additional cluster center is needed.  
The elbow method used by looking at a scree plot is still pretty subjective, but let's take a look to 
see how many cluster centers might be indicated.'''

'''once we have **fit** a kmeans model to some data in sklearn, there is a **score** method, 
which takes the data.  This score is an indication of how far the points are from the centroids.  
By fitting models for centroids from 1-10, and keeping track of the score and the number of centroids, 
you should be able to build a scree plot'''

# This plot should have the number of centroids on the x-axis, and the absolute value of the score result on the y-axis.


# Lets create a scree plot - Fit a kmeans model with changing k from 1-10
# Obtain the score for each model (take the absolute value)
# Plot the score against k

scores = []
for i in range(1,11):
    kmeans = KMeans(i)
    model = kmeans.fit(data)
    scr = model.score(data)  # gives score
    scores.append(abs(scr))   
    
centers = list(range(1,11))
plt.plot(centers, scores)
#plt.title('Scree Plot')
#plt.xlabel('Centers')
#plt.ylabel('Av. Distance from Centroid');

''' Using the scree plot, K value of 4 looks better'''
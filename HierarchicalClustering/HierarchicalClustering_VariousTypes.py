# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:50:17 2019
"""

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage # Import scipy's linkage function to conduct the clustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

''''''''''''''''1.Importing the Dataset'''''''''''''''''''''

iris = datasets.load_iris()

# Lets have a look on the data
print(iris.data[:10])
print(iris.target)

'''''''''''''''''''''2.Clustering'''''''''''''''''''''''''''

# Let's now use sklearn's AgglomerativeClustering to conduct the heirarchical clustering

# Hierarchical clustering
# Hierarchical clustering using Ward linkage, Ward is the default linkage algorithm, so let's start with that

# Create an instance of AgglomerativeClustering with the appropriate parameters
ward = AgglomerativeClustering(n_clusters=3)
# Make AgglomerativeClustering fit the dataset and predict the cluster labels
ward_pred = ward.fit_predict(iris.data)

# Hierarchical clustering using complete linkage
complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
complete_pred = complete.fit_predict(iris.data)

# Hierarchical clustering using average linkage
avg = AgglomerativeClustering(n_clusters=3, linkage='average')
avg_pred = avg.fit_predict(iris.data)

'''To determine which clustering result better matches the original labels of the samples, 
we can use adjusted_rand_score which is an external cluster validation index which results 
in a score between -1 and 1, where 1 means two clusterings are identical of how they grouped 
the samples in a dataset (regardless of what label is assigned to each cluster'''

# Calculate the adjusted Rand score for the ward, complete and average linkage clustering labels
ward_ar_score = adjusted_rand_score(iris.target, ward_pred)
complete_ar_score = adjusted_rand_score(iris.target, complete_pred)
avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

# Lets check which algorithm results in the higher Adjusted Rand Score?
print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)

'''''''''''''''''''3.The Effect of Normalization on Clustering'''''''''''''''''''''''''

# Let's take another look at the dataset'''
print(iris.data[:10])

'''Looking at this, we can see that the forth column has smaller values than the rest of the columns,
and so its variance counts for less in the clustering process (since clustering is based on distance). 
let us normalize the dataset so that each dimension lies between 0 and 1, so they have equal weight in 
the clustering process.
This is done by subtracting the minimum from each column then dividing the difference by the range.
sklearn provides us with a useful utility called preprocessing.normalize() that can do that for us'''

normalized_X = preprocessing.normalize(iris.data)
print(normalized_X[:10])

'''Now all the columns are in the range between 0 and 1. Would clustering the dataset 
after this transformation lead to a better clustering? (one that better matches the original labels of the samples)'''

ward = AgglomerativeClustering(n_clusters=3)
ward_pred = ward.fit_predict(normalized_X)

complete = AgglomerativeClustering(n_clusters=3, linkage="complete")
complete_pred = complete.fit_predict(normalized_X)

avg = AgglomerativeClustering(n_clusters=3, linkage="average")
avg_pred = avg.fit_predict(normalized_X)


ward_ar_score = adjusted_rand_score(iris.target, ward_pred)
complete_ar_score = adjusted_rand_score(iris.target, complete_pred)
avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)

''''''''''''''''''''''4.Dendrogram visualization with scipy'''''''''''''''''''''''''''

# Let's visualize the highest scoring clustering result.

'''To do that, we'll need to use Scipy's linkage function to perform the clusteirng again
so we can obtain the linkage matrix it will later use to visualize the hierarchy'''

# Specify the linkage type. Scipy accepts 'ward', 'complete', 'average', as well as other values
# Pick the one that resulted in the highest Adjusted Rand Score
linkage_type = 'ward'
linkage_matrix = linkage(normalized_X, linkage_type)

# Plot using scipy's dendrogram function
#plt.figure(figsize=(22,18))
# plot using 'dendrogram()'
dendrogram(linkage_matrix)
plt.show()

'''''''''''''''''''''''''''5.Visualization with Seaborn's clustermap'''''''''''''''''''''''''''''''''

'''The seaborn plotting library for python can plot a clustermap,
which is a detailed dendrogram which also visualizes the dataset in more detail. 
It conducts the clustering as well, so we only need to pass it the dataset and the
linkage type we want, and it will use scipy internally to conduct the clustering'''
 
sns.clustermap(normalized_X, figsize=(12,18), method=linkage_type, cmap='viridis')

# Expand figsize to a value like (18, 50) if you want the sample labels to be readable
# Draw back is that you'll need more scrolling to observe the dendrogram

plt.show()
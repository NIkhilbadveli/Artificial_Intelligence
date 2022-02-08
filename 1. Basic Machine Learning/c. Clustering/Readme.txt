Clustering comes under unsupervised learning, meaning the goal is to find natural patterns/clusters in the given data.

*** There are several models/algorithms for achieving this goal. One crucial thing to note here, is some models work best for some data.
The trick is to learn to identify which one is good at what scenario ***

For k-means clustering, the model is simply trying to find k mean data points such that the distance is minimum in the group.
For dbscan clustering, the model is
For gaussian clustering, the model is

There are other types of clustering as well, such as BIRCH, OPTICS, Mean shift, Affinity propagation etc.,

Check out this article to know more: https://machinelearningmastery.com/clustering-algorithms-with-python/
# https://medium.com/p/d40f2b34ae7e

# I'll try to use 3 different problems and fit the data for the above 3 types of clustering.
# I'll try to choose the data which has minimal pre-processing.
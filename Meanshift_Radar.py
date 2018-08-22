
# coding: utf-8

# In[ ]:

#import all libraries
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import *
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import *
from itertools import cycle


# In[ ]:

#get working directory
import os
os.getcwd()


# In[43]:

#read the data
X=pd.read_excel(r"Downloads\Radar_Dataset_397880.xlsx")
X


# In[44]:

#dropped NAs and reset index
X1=X.dropna()
X2= X1.reset_index(drop=True)
X2


# In[45]:

#Extracting timestamp column for clustering 
X3= X2.drop(['REL_SPEED',"NEW_TRACK","LAT_DIST"], 1)
X3.reset_index(drop=True)


# In[46]:

#Checking the length of extracted tiestamd data in X3
len(X3)


# In[ ]:

#http://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py


# In[47]:

#reshaping the sequence into 2D array because of the syntax requirements
#X4=np.reshape(X3, (-1, 2))
#X4
X4=X3.reset_index().values

X5= np.delete(X4, 0, 1) 
X5


# In[49]:

len(X5)


# In[50]:

#Estimating bandwidth
#note: this bandwidth is calculated considering all data ppoints as centroids atleast once and then narrowed
#down to the resultant band
bandwidth = estimate_bandwidth(X5, n_jobs=5)
bandwidth


# In[51]:

#fitting the data into clustering model based on the calculated bandwidth
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X5)



# In[52]:

#Finding out Cluster centers
labels = ms.labels_
cluster_centers = ms.cluster_centers_
cluster_centers


# In[53]:

#Getting number of clusters
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


# In[ ]:

labels_unique


# In[54]:

#Plot the results
plt.figure(1)
plt.clf()
 
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X5[my_members, 0], X5[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1],
             'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


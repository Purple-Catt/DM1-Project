import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


df = pd.read_csv("/content/drive/MyDrive/Università/Magistrale/Data Mining/dataset (missing + split)/clustering_dataset.csv") # read the data
df_train = df.drop(['Unnamed: 0'], axis=1) # drop the first column

# Hierarchical clustering
Z = linkage(df_train, method='ward', metric='euclidean')

# Dendrogram
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.show()

df_clusters = df_train.copy()

connectivity = kneighbors_graph(df_clusters, n_neighbors=100, include_self=False)
ward = AgglomerativeClustering(n_clusters=7, linkage='ward', metric='euclidean', connectivity=connectivity)
ward.fit(df_clusters)

cluster_labels = ward.labels_

# Plot the cluster centers
plt.figure(figsize=(20, 6))
n_clusters = 7
for i in range(n_clusters):
    plt.plot(range(0, len(df_train.columns)), df_train.iloc[cluster_labels == i].mean(), label='Cluster %s' % i, linewidth=3)

plt.xticks(range(0, len(df_train.columns)), list(df_train.columns))
plt.legend(bbox_to_anchor=(1, 1))
plt.grid(axis='y')
plt.show()

for i in range(len(df_clusters.columns)):
    for j in range(i + 1, len(df_clusters.columns)):
        sns.scatterplot(data=df_clusters, x=df_clusters.columns[i], y=df_clusters.columns[j], hue=ward.labels_, style=ward.labels_, palette="bright")
        plt.show()
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import *
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, BisectingKMeans
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
df = pd.read_csv("clustering_dataset.csv", index_col=0)
cl_col = ["duration_ms", "popularity", "danceability", "loudness", "speechiness", "acousticness",
          "instrumentalness", "liveness", "valence", "tempo"]


def kmean(df_scaled: pd.DataFrame = df):
    sse_list = []
    sil_list = []

    for k in range(2, 51):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=100, verbose=1)
        # kmeans = BisectingKMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=100, verbose=1)
        kmeans.fit(df_scaled)
        sse_list.append(kmeans.inertia_)
        sil_list.append(silhouette_score(df_scaled, kmeans.labels_))

    fig, axs = plt.subplots(2)  # 1 row, 2 columns

    sns.lineplot(x=range(2, 51), y=sse_list[:49], marker='o', ax=axs[0])
    axs[0].set(xlabel='k', ylabel='SSE')
    axs[0].plot([7], sse_list[5], "ro")
    sns.lineplot(x=range(2, 51), y=sil_list[:49], marker='o', ax=axs[1])
    axs[1].set(xlabel='k', ylabel='Silhouette')
    axs[1].plot([7], sil_list[5], "ro")
    plt.tight_layout()


def compare(df_scaled: pd.DataFrame = df):
    kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10, max_iter=100)
    kmeans.fit(df_scaled)
    bkmeans = BisectingKMeans(init='k-means++', n_clusters=7, n_init=10, max_iter=100)
    bkmeans.fit(df_scaled)
    bkmeans_ = KMeans(init=bkmeans.cluster_centers_, n_clusters=7, n_init="auto", max_iter=100)
    bkmeans_.fit(df_scaled)
    print("KMEANS")
    print('labels', np.unique(kmeans.labels_, return_counts=True))
    print('sse', kmeans.inertia_)
    print('silhouette', silhouette_score(df_scaled, kmeans.labels_))
    print("BISECT")
    print('labels', np.unique(bkmeans_.labels_, return_counts=True))
    print('sse', bkmeans_.inertia_)
    print('silhouette', silhouette_score(df_scaled, bkmeans_.labels_))

    plt.figure(figsize=(10, 3))

    for i in range(len(kmeans.cluster_centers_)):
        plt.plot(range(0, 10), bkmeans.cluster_centers_[i], label='Cluster %s' % i, linewidth=1, marker="o",
                 markersize=5)
    plt.xticks(range(0, 10), list(df_scaled.columns))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(axis='y')
    plt.xticks(rotation=45)

    df_clusters = df_scaled.copy()
    df_clusters['Labels'] = kmeans.labels_
    df_clusters_b = df_scaled.copy()
    df_clusters_b["Labels"] = bkmeans.labels_

    l = []
    for i in cl_col:
        for j in cl_col:
            if j in l:
                pass
            elif i == j:
                pass
            else:
                fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns
                sns.scatterplot(data=df_clusters,
                                x=i,
                                y=j,
                                hue=kmeans.labels_,
                                style=kmeans.labels_,
                                palette="bright",
                                legend=False,
                                ax=axs[0])
                axs[0].set_title("K-Means")
                sns.scatterplot(data=df_clusters_b,
                                x=i,
                                y=j,
                                hue=bkmeans.labels_,
                                style=bkmeans.labels_,
                                palette="bright",
                                legend=False,
                                ax=axs[1])
                axs[1].set_title("Bisecting K-Means")
                plt.show()
                plt.clf()
        l.append(i)


def db_scan():
    k = 4
    dist = pdist(df, "euclidean")
    dist = squareform(dist)
    kth_dist = []
    for d in dist:
        index_kth_distance = np.argsort(d)[k]
        kth_dist.append(d[index_kth_distance])
    plt.plot(range(0, len(kth_dist)), sorted(kth_dist))
    for eps in [0.65]:
        for minsamp in [4]:
            dbscan = DBSCAN(eps=eps, min_samples=minsamp, metric='euclidean')
            dbscan.fit(df)
            sns.scatterplot(data=df,
                            x="acousticness",
                            y="loudness",
                            hue=dbscan.labels_,
                            style=dbscan.labels_,
                            palette="bright",
                            legend="auto")
            plt.show()
            plt.clf()


connectivity = kneighbors_graph(df, n_neighbors=100, include_self=False)
ward = AgglomerativeClustering(n_clusters=7, linkage='complete', metric='euclidean', connectivity=connectivity)
ward.fit(df)

print(silhouette_score(df, ward.labels_))

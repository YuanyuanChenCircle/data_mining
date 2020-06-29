"""
    This file provides logic about clustering
"""
import json
import time
import warnings
import numpy as np
from sklearn import cluster


from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import operator


clustering_algorithms = {
        'KMeans': cluster.KMeans,
        'MiniBatchKMeans': cluster.MiniBatchKMeans,
        'AffinityPropagation': cluster.AffinityPropagation,
        'MeanShift': cluster.MeanShift,
        'SpectralClustering': cluster.SpectralClustering,
        'AgglomerativeClustering': cluster.AgglomerativeClustering,
        'DBSCAN': cluster.DBSCAN,
        'Birch': cluster.Birch,
}

clustering_method_name = {
        "KMeans": 'KMeans',
        "MiniBatchKMeans": "MiniBatchKMeans",
        "AffinityPropagation": "AffinityPropagation",
        # "GB": "GradientBoostingClassifier",
        "MeanShift": "MeanShift",
        "SpectralClustering" : "SpectralClustering",
        "AgglomerativeClustering" : "AgglomerativeClustering",
        "DBSCAN" : "DBSCAN",
        "Birch" : "Birch"
    }

def MyClustering(df, cluster_method, param=""):
    param_literal = json.loads(param)

    t_start = time.process_time()
    param_literal = json.loads(param)
    print(param_literal)
    # the cluter method and its param
    cur_cluster_method = clustering_algorithms[clustering_method_name[cluster_method]](**param_literal)


    # catch warnings related to kneighbors_graph
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
            category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
            category=UserWarning)
        cur_cluster_method.fit(df)

    t_end = time.process_time()
    t_diff = t_end-t_start
    stat = {'Clusatering_Algorithm': cluster_method,
              'train_time': str(t_diff)+"s"}

    if hasattr(cur_cluster_method, 'labels_'):
        y_pred = cur_cluster_method.labels_.astype(np.int)
    else:
        y_pred = cur_cluster_method.predict(df)

    new_df = df.copy(deep=True)
    new_df["clustering_result"] = y_pred
    # print("clustering_result")
    # print(df)
    # print("*"*10)
    # print(y_pred)



    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(df)
    X_pca = PCA(n_components=2).fit_transform(df)

    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(10, 5))

    print("#################")
    print(X_tsne.shape)
    print(X_tsne[:, 0].shape)
    # print(X_tsne[:,1])
    print(y_pred.shape)

    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, label="t-SNE")
    plt.legend()
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, label="PCA")
    plt.legend()
    plt.savefig('./polls/static/polls/images/digits_tsne-pca.png', dpi=120)
    # plt.show()

    return new_df, stat



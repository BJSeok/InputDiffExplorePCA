# silscore.py
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
# Author: Byoungjin Seok (bjseok@seoultech.ac.kr)
#

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def kmeans_clustering(data, num_clusters, n_init=10):
    kmeans = KMeans(n_clusters=num_clusters, n_init=n_init)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    return np.array(labels)


def calculate_silhouette(data, labels):
    silhouette_avg = silhouette_score(data, labels)
    
    return np.array(silhouette_avg)

def visualize_clusters(data, labels, title=None):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels.astype(np.float64), edgecolor='k')

    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()
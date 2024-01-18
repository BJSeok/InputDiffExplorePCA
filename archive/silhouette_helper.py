# silhouette_helper.py
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
# Author: Byoungjin Seok (bjseok@seoultech.ac.kr)
#

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

result_list = ['k-means clustering - Blocks Cryptset based on Random Number',
               'k-means clustering - Blocks Cryptset based on Ciphertext']

def cal_silscore(df_pca, num_clusters, algo):
    """Densenet model training

    Parameters
    ----------
    df_pca : dataframe
        Dataframe of PCA results
    algo : string
        Algorithm name

    Returns
    -------
    None

    """
    for i in range(len(df_pca)):
        fig = plt.figure(figsize=(16,9))
        ax = Axes3D(fig, elev=48, azim=134)
        est = [str(algo), KMeans(n_clusters=num_clusters)]
        est[1].fit(df_pca[i])
        labels = est[1].labels_
        ax.scatter(df_pca[i][:,0], df_pca[i][:,1], df_pca[i][:,2], c=labels.astype(np.float), edgecolor='k')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        sil_score = silhouette_score(df_pca[i], labels, metric='euclidean')
        plt.suptitle(str(algo) + ' ' + str(result_list[i]) + "\n" + "sil_score : " + str(sil_score))
        ax.dist = 12
        dirpath = "./silhouette_score/"

        try:
            if not(os.path.isdir(dirpath)):
                os.makedirs(os.path.join(dirpath))
        except OSError as e:
            if e.errno != errno.EXIST:
                print("Failed to create directory!!!!!")
            raise  

        plt.savefig(dirpath + algo.upper() + ' Silhouette_score - ' + result_list[i] + '.png', dpi=300)
        plt.show()

    return None
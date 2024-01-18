# pca_helper.py
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
# Author: Byoungjin Seok (bjseok@seoultech.ac.kr)
#

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
import numpy as np
from os import urandom
import csv
from datetime import datetime

def check_variance(dataframe, algo, title):
    """Calculate variance of dataframe

    To find the optimal dimension for drawing a graph,
    calculate variance and show result by graph

    Parameters
    ----------
    dataframe : dataframe
        Dataframe of block cipher algorithm dataset
    algo : string
        Algorithm name
    title : string
        Graph title

    Returns
    -------
    None

    """
    scaler = StandardScaler()
    pca = PCA()
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(dataframe)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.title(algo.upper() + ' Variance - ' + title)
    plt.xticks(features)

    dirpath = "./pca/variance_result/"

    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
    except OSError as e:
        if e.errno != errno.EXIST:
            print("Failed to create directory!!!!!")
        raise  

    plt.savefig(dirpath + algo.upper() + ' Variance - ' + title + '.png', dpi=300)
    plt.show()
    plt.close()
    
    return None             
    
# Make 3D
def decompose_3D(dataframe, case, algo, title, diff=None):
    """PCA of dataframe

    To find the optimal dimension for drawing a graph,
    calculate variance and show result by graph

    Parameters
    ----------
    dataframe : dataframe
        Dataframe of block cipher algorithm dataset
    case : string
        Case
    algo : string
        Algorithm name
    title : string
        Graph title
    diff : int
        Differencial

    Returns
    -------
    Filepath of PCA result

    """
    model = PCA(n_components=3)
    pca_features = model.fit_transform(dataframe)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xf = pca_features[:,0]
    yf = pca_features[:,1]
    zf = pca_features[:,2]
    ax.scatter(xf, yf, zf, s=0.1)
    plt.title(algo.upper() + ' PCA - ' + title)
    dirpath = "./pca/pca_result/"

    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
    except OSError as e:
        if e.errno != errno.EXIST:
            print("Failed to create directory!!!!!")
        raise  

    plt.savefig(dirpath + algo.upper() + ' PCA - ' + title + '.png', dpi=300)
    plt.show()
    plt.close()
            
    if diff==None:
        np.savetxt(dirpath + 'data_' + case + '.csv', pca_features, delimiter=',', fmt='%s')
    else:
        np.savetxt(dirpath + 'data_' + case + '_' + str('{:08x}'.format(diff)) + '.csv', pca_features, delimiter=',', fmt='%s')
    
    return dirpath + 'data_' + case + '.csv'
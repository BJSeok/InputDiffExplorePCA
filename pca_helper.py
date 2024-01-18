# pca_helper.py
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
# Author: Byoungjin Seok (bjseok@seoultech.ac.kr)
#

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def EigenValueDecomposition(dataset, alg=None, title=None, visualize_ratio='no'):
    scaler = StandardScaler()
    pca = PCA()
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(dataset)
    pca.fit_transform(dataset)
    
    if visualize_ratio == 'yes':
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_)
        plt.xlabel('features')
        plt.ylabel('variance')
        if alg is not None and title is not None:
            plt.title(alg.upper() + ' Variance - ' + title)
        plt.xticks(features)
        plt.show()
        plt.close()
    return pca.explained_variance_ratio_, pca.components_
    
def DimensionReduction(dataset, n_components = 3, alg=None, title=None):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(dataset)
    pca_results = pca.fit_transform(dataset)
    
    return pca_results

def Visualize3D(pca_results_3D, title=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xf = pca_results_3D[:,0]
    yf = pca_results_3D[:,1]
    zf = pca_results_3D[:,2]
    ax.scatter(xf, yf, zf, s=0.1, color='blue')
    
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()
# diffneural.py
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
# Author: Byoungjin Seok (bjseok@seoultech.ac.kr)
#

import os
import pandas as pd
import numpy as np

import datagen_helper
import train_distinguisher as td
import pca_helper as pca
import silhouette_helper as sil

algo = None
data_size = None
num_rounds = None

result_dict = {
    "df_blocks_random" : "Blocks Cryptset based on Random Number",
    "df_blocks_cipher" : "Blocks Cryptset based on Ciphertext"
}

def input_info():
    """Generate block cipher algorithm dataset and return dataframe dictionary of dataset

    Gnerate data according to user input.

    Parameters
    ----------
    None

    Returns
    -------
    df_dict : dictionary
        Dataframe dictionary of block cipher algorithm dataset

    """
    global algo
    global data_size
    global num_rounds
    algo = input("Crypto Algorithm : ")
    data_size = input("Data Size : ")
    num_rounds = input("Round : ")
    diff = input("Select diff (ex 0x00004000) : ")
    diff = diff[2:]
    
    filepath = datagen_helper.gen_data(algo, data_size, num_rounds, diff)
    
    df_dict = {}
    df_dict['df_blocks_random'] = pd.read_csv(filepath[0], header=None)
    df_dict['df_blocks_cipher'] = pd.read_csv(filepath[1], header=None)

    return df_dict
    
def show_variances(df_dict):
    """Show variance result of dataframe

    Parameters
    ----------
    df_dict : dictionary
        Dataframe dictionary of block cipher algorithm dataset

    Returns
    -------
    None

    """
    for dataframe_name in df_dict:
        pca.check_variance(df_dict[dataframe_name].loc[:, :len(df_dict[dataframe_name].columns) - 2], algo, result_dict[dataframe_name])
    
    return None
    
def show_pcas(df_dict):
    """Show variance result of dataframe

    Parameters
    ----------
    df_dict : dictionary
        Dataframe dictionary of block cipher algorithm dataset

    Returns
    -------
    pca_filepath : list
        Result csv filepaths of PCA

    """
    pca_filepath = []
    for dataframe_name in df_dict:
        pca_filepath.append(pca.decompose_3D(df_dict[dataframe_name].loc[:, :len(df_dict[dataframe_name].columns) - 2], 'pca_' + str(dataframe_name[3:]), algo, result_dict[dataframe_name]))

    return pca_filepath

def resnet_blocks(df_dict):
    """Show resnet training result of blocks dataset dataframe

    Compose residual neural network according to user input.

    Parameters
    ----------
    df_dict : dictionary
        Dataframe dictionary of block cipher algorithm dataset

    Returns
    -------
    None

    """
    batch_size = int(input("Batch Sizes : "))
    num_filters = int(input("Filters : "))
    kernel_size = int(input("Kernel Sizes : "))
    num_epochs = int(input("Epochs : "))
    depth = int(input("Depth : "))
    regularizer = float(input("Regularizer : "))
    
    # training set and validations set
    blocks_trainset, blocks_validset = td.df2TrainValidSet(df_dict['df_blocks_random'], df_dict['df_blocks_cipher'])

    blocks_Y = td.df2ndarray(blocks_trainset[len(blocks_trainset.columns) - 1])
    blocks_Y_eval = td.df2ndarray(blocks_validset[len(blocks_validset.columns) - 1])
    
    del blocks_trainset[len(blocks_trainset.columns) - 1]
    del blocks_validset[len(blocks_validset.columns) - 1]

    blocks_X = td.df2ndarray(blocks_trainset)
    blocks_X_eval = td.df2ndarray(blocks_validset)
    
    td.train_resnet_distinguisher(blocks_X, blocks_Y, blocks_X_eval, blocks_Y_eval, algo, num_rounds, batch_size, num_filters, kernel_size, num_epochs, depth, regularizer)
        
    return None

def fcnn_blocks(df_dict):
    """Show dfcnn training result of blocks dataset dataframe

    Compose deep neural network according to user input.

    Parameters
    ----------
    df_dict : dictionary
        Dataframe dictionary of block cipher algorithm dataset

    Returns
    -------
    None

    """
    batch_size = int(input("Batch Sizes : "))
    num_epochs = int(input("Epochs : "))
    num_layers = int(input("Layers : "))
    
    # training set and validations set
    blocks_trainset, blocks_validset = td.df2TrainValidSet(df_dict['df_blocks_random'], df_dict['df_blocks_cipher'])

    blocks_Y = td.df2ndarray(blocks_trainset[len(blocks_trainset.columns) - 1])
    blocks_Y_eval = td.df2ndarray(blocks_validset[len(blocks_validset.columns) - 1])
    
    del blocks_trainset[len(blocks_trainset.columns) - 1]
    del blocks_validset[len(blocks_validset.columns) - 1]

    blocks_X = td.df2ndarray(blocks_trainset)
    blocks_X_eval = td.df2ndarray(blocks_validset)
    
    td.train_fcnn_distinguisher(blocks_X, blocks_Y, blocks_X_eval, blocks_Y_eval, algo, num_rounds, batch_size, num_epochs, num_layers)
    
    return None

def densenet_blocks(df_dict):
    """Show densenet training result of blocks dataset dataframe

    Compose densely connected convolutional network according to user input.

    Parameters
    ----------
    df_dict : dictionary
        Dataframe dictionary of block cipher algorithm dataset

    Returns
    -------
    None

    """
    batch_size = int(input("Batch Sizes : "))
    num_filters = int(input("Filters : "))
    kernel_size = int(input("Kernel Sizes : "))
    num_epochs = int(input("Epochs : "))
    depth = int(input("Depth : "))
    growth_rate = int(input("Growth Rate : "))
    regularizer = float(input("Regularizer : "))
    
    # training set and validations set
    blocks_trainset, blocks_validset = td.df2TrainValidSet(df_dict['df_blocks_random'], df_dict['df_blocks_cipher'])

    blocks_Y = td.df2ndarray(blocks_trainset[len(blocks_trainset.columns) - 1])
    blocks_Y_eval = td.df2ndarray(blocks_validset[len(blocks_validset.columns) - 1])
    
    del blocks_trainset[len(blocks_trainset.columns) - 1]
    del blocks_validset[len(blocks_validset.columns) - 1]

    blocks_X = td.df2ndarray(blocks_trainset)
    blocks_X_eval = td.df2ndarray(blocks_validset)
    
    td.train_densenet_distinguisher(blocks_X, blocks_Y, blocks_X_eval, blocks_Y_eval, algo, num_rounds, batch_size, num_filters, kernel_size, num_epochs, depth, growth_rate, regularizer)
        
    return None

def show_silscore(pca_filepath):
    """Show silhouette score result of K-menas clustering

    Parameters
    ----------
    pca_filepath : List
        Filepath list of PCA results

    Returns
    -------
    None

    """
    num_clusters = int(input("Clusters : "))
    df_pca = []
    df_pca.append(pd.read_csv(pca_filepath[0], header=None))
    df_pca.append(pd.read_csv(pca_filepath[1], header=None))
    
    df_pca[0] = np.array(df_pca[0].values.tolist())
    df_pca[1] = np.array(df_pca[1].values.tolist())
    
    sil.cal_silscore(df_pca, num_clusters, algo)

    return None
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:42:08 2023

@author: LuisJa
"""

'''
    This file contains the dataloader function. By giving it a ASCII point cloud file name, 
    it computes neighbors, cuts the point cloud in small patches 
    and return a DataLoader object from the torch_geometric library, as well as the weights of each class.
    The less predominant a class is, the bigger the weight. Weights do not sum up to 1.
    Note that dataloader can (and generally) returns batches of varying number of points
    
'''


import torch
from torch import Tensor, LongTensor

import torch_geometric
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader, DataListLoader
except:
    from torch_geometric.data import DataLoader,  DataListLoader
from torch_geometric.nn.pool import nearest

#import other libraries
import numpy as np 
import os
import glob
import pandas as pd




def get_dataloader(data_root, filename = None, pos_index = [0,1,2], label_index = None, features_index = None, sample_size=6000, batch_size=2, log=False):
    """
    

    Parameters
    ----------
    data_root : string
        directory where to find the file
    filename : string
        name of the file
    pos_index : list[], optional
        list of indexes of the positions of points XYZ (XY should also work I think ?). The default is [0,1,2].
    label_index : int, optional
        index of the label columns. The default is None.
    features_index : TYPE, optional
        list of indexes of the features. The default is None.
    sample_size : int, optional
        numbers of points in a sample. The default is 6000.
    batch_size : int, optional
        number of samples in a batch. The default is 2.
    log : bool, optional
        Wether to print or not informations on each batch. The default is False

    Returns
    -------
    dataloader : torch_geometric.Data.Dataloader
        dataloader which can be itered through a for loop or next(iter(loader)) to get batches.
    weights : Tensor, dim 1xC, C being number of classes.
        Tensor containing weights of each class. The more elements in a class, the smaller the weight.

    """
    
    
    #Convert features index to int
    features_index = [int(f) for f in features_index] if features_index is not None else None
    
    
    #Read header of the file to get columns names as well as features and labels used
    partitionnings = []
    if filename is not None:
        files = [os.path.join(data_root,filename)]
    else:
        files = glob.glob(data_root + '/**/*.txt', recursive=True)
        
    for i,file in enumerate(files): 
        with open(file) as f:
            first_line = f.readline().split(" ")
            print("all features: ", [ (col,i) for col,i in enumerate(first_line)])                                      #print all available features
            print("name of the classification column: ", first_line[label_index] if label_index != None else None)                       #print labels used
            print("selected features: ", [first_line[i] for i in features_index] if features_index != None else None)  #print features used
            print("\n \n")
            
        
        #load data
        arr = pd.read_csv(file, sep=" ")
        ids = arr.index.to_numpy()
        arr = arr.to_numpy() #using pandas for I/O cause waaaaay faster
        arr[np.isnan(arr)] = 0          #features can be nan
        position = arr[:,pos_index]     #load XYZ, dim Nx3
        
        
        #Handle labels and features index when existing
        #convert label to LongTensor (int), dim : Nx1, N being number of points
        labels = Tensor(arr[:,label_index]).type(LongTensor) if label_index is not None else None 
        #convert features to Tensor, dim : Nxf, N being number of points, f number of features
        features = Tensor(arr[:,features_index]) if features_index is not None else None   
        
        #Create Data
        data = Data(pos = Tensor(position),  x=features, y=labels, ids=LongTensor(ids)) #convert data to torch_geometric.Data object 
    
        def random_iter(data,sample_size=6000):
            """
            Recursive algorithm which partitions data object in objects of size 'sample_size' at most
    
            Parameters
            ----------
            data : data object
                torch_geometric data object.
            sample_size : int, optional
                Maximum number of points in a continuous patch. The default is 6000.
    
            Returns
            -------
            data : list(torch_geometric.Data objects)
                A list of data objects which size are less than sample_size.
    
            """
            
            patch_list = []
            cloud  = data.pos
            #Initialisation case
            if cloud.shape[0] < sample_size:
                return [data]
            #recursive case
            else:
                cluster_index = nearest(cloud[:,:3], cloud[np.random.choice(cloud.shape[0],replace=False, size=2)][:,:3])

                datas = []
                for i in np.unique(cluster_index):
                    d = Data()
                    for key in data.keys:
                        d[key] = data[key][cluster_index == i]
                    datas.append(d)
                    
                for sub_data in datas:
                    patch_list += random_iter(sub_data, sample_size)

                return patch_list    
       
            
            return data
    

        partitionning = random_iter(data, sample_size)
    
        for batch in partitionning:
            if torch_geometric.__version__ != '1.7.1':   
                batch.batch = i * torch.ones(batch.pos.shape[0])
            partitionnings.append(batch)
    
    
    #Convert list(Data) to DataLoader
    dataloader = DataLoader(partitionnings, batch_size=batch_size, shuffle=True)
    
    #Print Informations
    if log:
        data = next(iter(dataloader))  # Get the first graph object.

    
    #Compute weights if labels exist
    weights = None
    if labels != None:
        weights = Tensor([1 /i for i in np.unique(labels[labels != -1], return_counts=True)[1] / len(labels) ]) # remove unclassified points (-1) from the class weights
    
    return dataloader, weights


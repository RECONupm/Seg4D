# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:42:08 2023

@author: LuisJa
"""

#%% LIBRARIES
import torch
from torch import Tensor, LongTensor
import torch_geometric
from torch_geometric.nn import radius_graph
import torch_geometric.transforms as T
from torch_geometric.nn import DataParallel
import argparse
import numpy as np 
import platform
import time
import os
import pandas as pd
from dataloader import get_dataloader
from point_transformer_segmentation import Net


# parsing command line arguments. Example : python inference.py --input cloud.txt --output result.txt --model model.pt
parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str, help='path and name of the point cloud to process')
parser.add_argument('--output', type=str, help='where to save the processed point cloud')
parser.add_argument('--features', type=str, help='features index to use separated by comma. Ex : 1,2,4. Default : None', default=None)
parser.add_argument('--model', type=str, help='model to use')
parser.add_argument('--size', type=int, help='numbers of points processed at the same time. Higher values need more memory. Default=48000', default=48000)
args = parser.parse_args()

feat = args.features.split(',') if args.features is not None else None
data_root, cloud_filename = os.path.split(args.input)
final_loader, _ = get_dataloader(data_root, cloud_filename,  features_index = feat, sample_size = args.size // 4, batch_size=4)


#Try to work on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() == 1:
    print("working with the GPU")
if torch.cuda.device_count() > 1:
    print("working with several GPUs")
if device.type == 'cpu':
    print("working with the CPU")
  
#load the architecture from pt file.
architecture = torch.load(args.model)

#Get number of features and number of classes automatically by looking ar architecture dimensions
NUM_FEATURES = architecture[[*architecture.keys()][ 0]].shape[1]
NUM_CLASSES  = architecture[[*architecture.keys()][-1]].shape[0]

print("number of features detected : ", NUM_FEATURES)
print("number of classes detected : ", NUM_CLASSES)

#Create the model
model = Net(3 + NUM_FEATURES, NUM_CLASSES, dim_model=[
            32, 64, 128, 256, 512], k=16)

from torch.nn import Linear as Lin
model.mlp_input[0][0] = Lin(NUM_FEATURES,32)
model = DataParallel(model)
model.load_state_dict(architecture)

model = model.to(device)

transform = T.Compose([
])


def final(loader):
    """
    

    Parameters
    ----------
    loader : torch_geometric.Data.Dataloader
        data loader of cloud to classify.

    Returns
    -------
    cloud_output : numpy.array, dim N x (2 + C)
        Numpy array, contains id_points, 
    """
    model.eval()
    cloud_output = []
    with torch.no_grad():
        for i,data in enumerate(loader):
            start_time = time.time()
        
            data_tf = transform(data)
            data_tf = data_tf.to(device)
            out = model(data_tf.to_data_list())  # Perform a single forward pass.
            
            scores = out
            data = data.cpu()
            labels = scores.argmax(dim=1).type(torch.LongTensor)
            scores = scores.cpu()     
            
            ids = data.ids.reshape((-1,1))
            tupl = [ids, labels.reshape((-1,1)), scores]
            result = np.concatenate(tupl,axis=1)
            cloud_output.append(result)
        
            end_time = time.time()
            
        
    cloud_output = np.vstack(cloud_output) #convert to array
    
    return cloud_output

result = final(final_loader)
sorted_result = result[ np.argsort( result[:,0])]

arr = pd.read_csv(args.input, sep=" ")

formats = []
for c in arr.columns:
    i = arr[c].first_valid_index()
    liste = str( arr[c][i] ).split('.')
    prec = len( liste[1] ) if len(liste) != 1 else 0
    formats.append('%1.' + str(prec) + 'f')

arr['prediction'] = sorted_result[:,1]
formats.append('%1d')
for i in range(2, sorted_result.shape[1]):
    arr['class_' + str(i-1) + '_score'] = sorted_result[:,i]
    formats.append('%1.3f')

np.savetxt(args.output, arr.values, fmt=formats, delimiter=' ',comments='', header=' '.join(arr.columns))


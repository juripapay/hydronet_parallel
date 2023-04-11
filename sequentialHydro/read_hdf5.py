#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# python read_hdf5.py
# Read hdf5 file (for examle qm9.hdf5) and create a .ps file for training/validatio/test
# in the ratio of 80/10/10 of samples
# 



'''
For running the hydronet benchmark we need to install dependencies with specific versions, otherwise
the application will not work. The dependencies can be installed by the following commands:

1)      Create conda environment
conda create --name hydronet2 python=3.8

2)      Activate conda environment
activate conda hydronet2

3)      Installing pytorch:
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge

4)      conda install pyg -c pyg

5)      conda install -c conda-forge tensorboard ase fair-research-login h5py tqdm

6)      conda install -c conda-forge gdown

7) pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

'''

import time
import yaml
import torch
import math
from pathlib import Path
import os, sys
import json, csv
import argparse, shutil
import logging
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import DataParallel
from tqdm import tqdm
import datetime



# Dataloader dependencies
import os.path as op
from torch_geometric.data import DataListLoader, DataLoader
from torch.utils.data import ConcatDataset
#from utils.datasets import PrepackedDataset


# Data loading
import h5py
import random
from torch_geometric.data import DataListLoader, DataLoader, InMemoryDataset, Data, extract_zip, download_url


# Libraries required for inference operation
import os.path as op
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt


#####
# init_dataloader
def init_dataloader(actionStr, split = '00', shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """
    pin_memory = True

    train_data = []
    val_data = []
    test_data = []
    
    hdf5_file = os.path.join('/home/pearl061/sciml_bench/datasets/hydronet_ds1','min.hdf5')
    dataset = h5py.File(hdf5_file, "r")

    # dataset info
    dataset_keys = list(dataset.keys())
    pos_data = dataset['pos']
    print("dataset_keys:", dataset_keys)
    print("pos_data[0]):", pos_data[0])
    cluster_size = 5
    x = torch.from_numpy(dataset['x'][0][:cluster_size])
   
    index=10
    dataset_size = int(dataset['size'].shape[0])
    indexList = []
    
    for i in range(0, dataset_size):
        indexList.append(i)
    # shuffle list
    shuffledList = random.shuffle(indexList)

    # Split up indexList into train, val and test in ratio of 80:10:10
    trainLen = int(dataset_size*0.8)
    valLen = int(dataset_size*0.1)
    testLen = valLen
    trainIndices = indexList[0:trainLen-1]
    valIndices = indexList[trainLen:trainLen+valLen-1]
    testIndices = indexList[trainLen+valLen:]
    print('len(trainIndices): ', len(trainIndices))
    
    if(actionStr =='train'):
        start_load = datetime.datetime.now()
        trainData = load_data(trainIndices, dataset)
        torch.save(trainData, 'min_trainData.pt')
        print(f'data load time:{datetime.datetime.now() - start_load}' )

        valData = load_data(valIndices, dataset)
        torch.save(valData, 'min_valData.pt')
        train_data.append(trainData)
        val_data.append(valData)
        # no shuffle
        train_loader = DataLoader(ConcatDataset(train_data), batch_size=128, shuffle=shuffle, pin_memory=pin_memory)
        val_loader = DataLoader(ConcatDataset(val_data), batch_size=128, shuffle=shuffle, pin_memory=pin_memory)
        
        return train_loader, val_loader

    if(actionStr =='test'):
        testData = load_data(testIndices, dataset)
        torch.save(testData, 'min_testData.pt')
        test_data.append(testData)
        test_loader = DataLoader(ConcatDataset(test_data), batch_size=128, shuffle=shuffle, pin_memory=pin_memory)
        return test_loader, testData

# load_data
def load_data(listIndices, dataset):
    logging.info("Loading cached data from disk...")
   
    data_list = []
    for i in range(len(listIndices)):
        index = listIndices[i]
        cluster_size = dataset["size"][index][0]

        z = torch.from_numpy(dataset["z"][index][:cluster_size])
        x = torch.from_numpy(dataset["x"][index][:cluster_size])
        pos = torch.from_numpy(dataset["pos"][index][:cluster_size])
        pos.requires_grad = True
        y = torch.from_numpy(dataset["y"][index])
        size = torch.from_numpy(dataset["size"][index])
        data = Data(x=x, z=z, pos=pos, y=y, size=size)
        data_list.append(data) 
    return data_list

#
# python read_hdf5.py
#
def main():
    # load data
    start_load = datetime.datetime.now()

    train_loader, val_loader = init_dataloader('train')
    train_loader, val_loader = init_dataloader('test')

    load_time = datetime.datetime.now() - start_load
    logging.info(f'Data load time: {load_time}')


if __name__ == "__main__":

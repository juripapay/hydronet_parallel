import os
import torch
import os.path as op
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from utils.datasets import PrepackedDataset
import torch.distributed as dist
import sys

class HydronetDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        self.dataset = torch.load(file_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


# init_dataloader_from_file
def init_dataloader_from_file(args, actionStr, split = '00', shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """
    pin_memory = False if args.train_forces else True

    #train_data = []
    #val_data = []
    #test_data = []
    
    # Read input file from args
    inputData = args.inputFile
    onlyFileName = inputData.split(".")[0]

    path1 = os.path.join(os.path.expanduser("~"), args.datadir)
    train_file = os.path.join(path1, onlyFileName+"_trainData.pt")
    val_file = os.path.join(path1, onlyFileName+"_valData.pt")
    test_file = os.path.join(path1, onlyFileName+"_testData.pt")

    if(actionStr =='train'):
        train_data = HydronetDataset(train_file)
        val_data = HydronetDataset(val_file)
        return train_data, val_data

    if(actionStr =='test'):
        testData = torch.load(test_file)
        return test_data

#
def init_dataloader(args, local_batch_size):
    """
    Returns train, val, and list of examine loaders
    """
    pin_memory = False if args.train_forces else True

    if not isinstance(args.datasets, list):
        args.datasets = [args.datasets]

    #train_data, val_data = init_dataloader_from_file(args,"train")
    trainData, valData = init_dataloader_from_file(args,"train")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainData, shuffle=True, drop_last=True)
    train_loader = DataLoader(trainData, sampler=train_sampler, batch_size=local_batch_size, shuffle=False, pin_memory=pin_memory)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(valData, shuffle=True, drop_last=True)
    val_loader = DataLoader(valData, sampler=val_sampler, batch_size=local_batch_size, shuffle=False, pin_memory=pin_memory)

    print("Train data size {:5d}".format(len(trainData)), flush=True)
    print("train_loader size {:5d}".format(len(train_loader.dataset)), flush = True)
    print("val_loader size {:5d}".format(len(val_loader.dataset)), flush = True)

    return train_loader, val_loader, train_sampler


def test_dataloader(args, 
                    dataset,
                    split = '00'
                    ):

    dataset = PrepackedDataset(None, 
                               op.join(args.savedir,f'split_{split}_{dataset}.npz'), 
                               dataset, 
                               directory=args.datadir)
    data = dataset.load_data('test')
    
    batch_size = args.batch_size if len(data) > args.batch_size else len(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return loader



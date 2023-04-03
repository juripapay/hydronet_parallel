import os
import torch
import os.path as op
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from utils.datasets import PrepackedDataset
import torch.distributed as dist
import sys

# This class was suggested by Jesun
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, train_file):
        self.dataset = torch.load(train_file) # assuming torch.load returns a list, initially self.dataset = []

    def __len__(self):
        # return total dataset size
        return len(self.z) # or any other way to retrieve the length of the total dataset in the new format

    def __getitem__(self, index):
        # return each batch element
        return self.x[index], self.z[index], self.pos[index], self.y[index], self.f[index], self.size[index]


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
        # load .pt files
        train_data = torch.load(train_file)
        # train_data = CustomDataset(train_file)
        val_data = torch.load(val_file)
        # val_data = CustomDataset(val_file)
        return train_data, val_data

    if(actionStr =='test'):
        testData = torch.load(test_file)
        return test_data

#
def init_dataloader(args,
                    ngpus_per_node,
                    split = '00', 
                    shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """
    pin_memory = False if args.train_forces else True

    if not isinstance(args.datasets, list):
        args.datasets = [args.datasets]

    train_data = []
    val_data = []
    examine_data = []
    
    #train_data, val_data = init_dataloader_from_file(args,"train")
    trainData, valData = init_dataloader_from_file(args,"train")
    
    train_data.append(trainData)
    val_data.append(valData)
    # no shuffle
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

    train_loader = DataLoader(ConcatDataset(train_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
    
    val_loader = DataLoader(ConcatDataset(val_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)


    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    

    '''
    train_loader = DataLoader(train_data, 
                              batch_size=int(args.batch_size), 
                              shuffle=(train_sampler is None),
                              num_workers=1 , 
                              pin_memory=pin_memory,
                              sampler=train_sampler, 
                              drop_last=True)

    print("train_loader size {:5d}".format(len(train_loader.dataset)), flush = True)

    val_loader = DataLoader(val_data, 
                            batch_size=int(args.batch_size), 
                            shuffle=False,
                            num_workers=1,
                            pin_memory=pin_memory,
                            drop_last=True)

    '''
    print("Train data size {:5d}".format(len(train_data)), flush=True)
    print("train_loader size {:5d}".format(len(train_loader.dataset)), flush = True)
    print("val_loader size {:5d}".format(len(val_loader.dataset)), flush = True)

    # ngpus_per_node
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



import os
import time
import torch
import os.path as op
from torch_geometric.data import DataLoader, Dataset, Data
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Subset
import torch.distributed as dist
import h5py
from pathlib import Path

class HydronetDataset(Dataset):

    def __init__(self, file_name):
        self.file_handle = h5py.File(file_name)

    def __len__(self):
        return self.file_handle['x'].shape[0]

    def __getitem__(self, index):
        size = torch.from_numpy(self.file_handle['size'][index])
        x = torch.from_numpy(self.file_handle['x'][index][:size])
        z = torch.from_numpy(self.file_handle['z'][index][:size])
        pos = torch.from_numpy(self.file_handle['pos'][index][:size])
        y = torch.from_numpy(self.file_handle['y'][index])
        data = Data(x=x, y=y, z=z, pos=pos, size=size)
        return data

    def __del__(self):
        self.file_handle.close()


# init_dataloader_from_file
def init_dataloader_from_file(args, actionStr, split = '00', shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """
    path = os.path.join(os.path.expanduser("~"), args.datadir)
    file_name = Path(path) / 'min.hdf5'
    dataset = HydronetDataset(file_name)
    
    # Split data 80:10:10
    n = len(dataset)
    indices = torch.arange(n)
    train_size = int(n*0.8)
    val_size = int(n*0.1)

    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[train_size:train_size+val_size])
    test_dataset = Subset(dataset, indices[train_size+val_size:])

    if(actionStr =='train'):
        return train_dataset, val_dataset

    if(actionStr =='test'):
        return test_dataset

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
    train_sampler = BatchSampler(train_sampler, batch_size=local_batch_size, drop_last=True)
    train_loader = DataLoader(trainData, batch_sampler=train_sampler, pin_memory=pin_memory, num_workers=2)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(valData, shuffle=True, drop_last=True)
    val_sampler = BatchSampler(val_sampler, batch_size=local_batch_size, drop_last=True)
    val_loader = DataLoader(valData, batch_sampler=val_sampler, pin_memory=pin_memory, num_workers=2)

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



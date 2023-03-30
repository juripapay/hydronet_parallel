'''
The DDP version of the code can be found in the schnet_ddp branch of:
https://github.com/jenna1701/hydronet/tree/schnet_ddp/challenge-1.5

For download use:
git clone --branch schnet_ddp https://github.com/jenna1701/hydronet/


Developed by: Firoz, Jesun S <jesun.firoz@pnnl.gov>

Dependencies:
-------------
conda create --name hydronet2 python=3.8
activate conda hydronet2
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install pyg -c pyg
conda install -c conda-forge tensorboard ase fair-research-login h5py tqdm
conda install -c conda-forge gdown
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

Run the code:
-------------
python train_direct.py --savedir './test_train' --args 'train_args.json'

'''

# NEED TO: conda install tensorboard
import os, sys
import torch
import shutil
import logging
import json 
import csv
import argparse
import time
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm
from utils import data_ddp, models, train, train_ddp, models_ddp, eval, split, hooks

# def init_setup(args):
def main(args):
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    dist.init_process_group(backend="nccl", init_method='env://')
    rank = int(os.environ["LOCAL_RANK"])  # dist.get_rank()
    print(f"Start running SchNet on rank {rank}.")
    ngpus_per_node = torch.cuda.device_count()
    device_id = rank # % torch.cuda.device_count()

    torch.cuda.set_device(device_id)

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )
    print(f'num_gpus: {ngpus_per_node}')
    print("Running the DDP model")
    
    ######## SET UP ########
    # create directory to store training results
    if device_id == 0:
        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)
            os.mkdir(os.path.join(args.savedir,'tensorboard')) 
        else:
            logging.warning(f'{args.savedir} is already a directory, either delete or choose new SAVEDIR')
            sys.exit()
    
        # copy args file to training folder
        shutil.copy(args.args, os.path.join(args.savedir, 'args.json'))

    dist.barrier()
    
    # read in args
    savedir = args.savedir
    with open(args.args) as f:
        args_dict = json.load(f)
        args = argparse.Namespace(**args_dict)
    args.savedir = savedir
    
    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device ", device)
    logging.info(f'model will be trained on {device}')

    torch.manual_seed(12345)
    net = models_ddp.load_model_ddp(args, device_id, device)

    # criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.8, min_lr=0.000001)
    
    # load datasets/dataloaders
    train_loader, val_loader, train_sampler = data_ddp.init_dataloader(args, ngpus_per_node)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
        
    start.record()
   
    for epoch in range(args.n_epochs):
        net.train()
        # let all processes sync up before starting with a new epoch of training
        dist.barrier()
        # DistributedSampler deterministically shuffle data
        # by seting random seed be current number epoch
        # so if do not call set_epoch when start of one epoch
        # the order of shuffled data will be always same
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        
        print("type(train_loader): ", type(train_loader))
        train_loss = train_ddp.train_energy_only_ddp(args, device_id, 
                                                     net, train_loader, optimizer, 
                                                     device)
        print(">>>>>> After train loss >>>>>")
        sys.exit()

        dist.barrier()
        val_loss = train_ddp.get_pred_eloss_ddp(args, device_id, net, 
                                            val_loader, optimizer, device)
        print("-" * 89, flush=True)
        print("| end of epoch {:3d} | time: {:5.2f}s |  valid loss {:5.4f} |".format(
                epoch, (time.time() - epoch_start_time), val_loss), flush=True)
        scheduler.step(val_loss) 
        dist.barrier()

    end.record()
    dist.destroy_process_group()
    # Waits for everything to finish running
    torch.cuda.synchronize()
        
    print(f'Elapsed time in miliseconds {start.elapsed_time(end)}')  # milliseconds

if __name__ == '__main__':
    # import path arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, required=True, help='Directory to save training results')
    parser.add_argument('--args', type=str, required=True, help='Path to training arguments')

    # # This is passed in via launch.py
    # parser.add_argument("--local_rank", type=int, default=0)
    # # This needs to be explicitly passed in
    # parser.add_argument("--local_world_size", type=int, default=1)

    args = parser.parse_args()
    main(args)

    # init_setup(args)
    # train_dataloader, val_dataloader = load_data(args)

    # world_size = torch.cuda.device_count()
    # print('Let\'s use', world_size, 'GPUs using DistributedDataParallel!')
    # mp.spawn(run, args=args, nprocs=world_size, join=True)
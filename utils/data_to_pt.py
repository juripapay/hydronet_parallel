import h5py
import torch
from pathlib import Path
from torch_geometric.data import DataLoader, Dataset, Data

def main():
    input_file = Path('~/sciml_bench/datasets/hydronet_ds1/qm9.hdf5')
    input_file = input_file.expanduser()
    with h5py.File(input_file) as f:
        n = f['x'].shape[0]
        for index in range(n):
            size = torch.from_numpy(f['size'][index])
            x = torch.from_numpy(f['x'][index][:size])
            z = torch.from_numpy(f['z'][index][:size])
            pos = torch.from_numpy(f['pos'][index][:size])
            y = torch.from_numpy(f['y'][index])
            data = Data(x=x, y=y, z=z, pos=pos, size=size)

            torch.save(data, f'qm9_data/data_{index}.pt')
          
if __name__ == "__main__":
    main()
import os.path as osp
import glob

import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class SUEPV1(Dataset):
    r'''
        input x0: (PF candidates)

        # truth targets = event class (S vs B)
    '''

    url = '/dummy/'

    def __init__(self, root, obj='PFCand', coords='cyl', transform=None, ratio=False):
        super(SUEPV1, self).__init__(root, transform)
        
        self.strides = [0]
        self.ratio = ratio
        self.obj = obj
        self.coords = coords
        self.calculate_offsets()

    def calculate_offsets(self):
        for path in self.raw_paths:
            with h5py.File(path, 'r') as f:
                self.strides.append(f['{}_{}'.format(self.obj, self.coords)].shape[0])
        self.strides = np.cumsum(self.strides)

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return self.strides[-1]

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.hdf5')))
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        file_idx = np.searchsorted(self.strides, idx) - 1
        idx_in_file = idx - self.strides[max(0, file_idx)] - 1
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        edge_index = torch.empty((2,0), dtype=torch.long)
        with h5py.File(self.raw_paths[file_idx],'r') as f:
            
            Nlc = np.count_nonzero(f['{}_{}'.format(self.obj, self.coords)][idx_in_file,:,0])

            # convert to torch
            x = torch.from_numpy(f['{}_{}'.format(self.obj, self.coords)][idx_in_file,:][None])
            x_pf = torch.from_numpy(f['{}_{}'.format(self.obj, self.coords)][idx_in_file,:Nlc,:]).float()

            # targets
            # y = torch.from_numpy(np.asarray(f['target'][idx_in_file]))
            if 'miniaod' in self.raw_paths[file_idx] or 'nanoaod' in self.raw_paths[file_idx]:
                y = torch.from_numpy(np.asarray(1.0))
            else:
                y = torch.from_numpy(np.asarray(0.0))
                
            # event level info
            S1 = torch.from_numpy(np.asarray(f['event_feat'][idx_in_file,2]))

            return Data(x=x, edge_index=edge_index, y=y,
                        x_pf=x_pf,
                        S1=S1)

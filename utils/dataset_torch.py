import os
import torch
from torch.utils.data import Dataset
import h5py


class LoadDataset(Dataset):
    """
    Dataset of random geometric figures composed by cylinders and spheres in different scan angles
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = os.listdir(self.folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        filename = os.path.join(self.folder, self.images[item])
        hf = h5py.File(filename, 'r')
        phase = torch.tensor(hf.get('phase')).permute(-1, 0, 1, 2)
        chi = torch.tensor(hf.get('chi')).permute(-1, 0, 1, 2)
        hf.close()
        sample = {'phase': phase, 'chi': chi}
        if self.transform:
            sample = self.transform(sample)
        return sample

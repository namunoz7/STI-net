import os
import torch
import h5py
from torch.utils.data import Dataset


class GeometricFiguresDataset(Dataset):
    """
    Dataset of random geometric figures composed by cylinders and spheres in different scan angles
    """
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.data_images = os.listdir(self.dataset_dir)

    def __len__(self):
        return len(self.data_images)

    def __getitem__(self, item):
        filename = os.path.join(self.dataset_dir, self.data_images[item])
        with h5py.File(filename, 'r') as hf:
            phase = hf.get('phase')
            phase = torch.tensor(phase).permute(-1, 0, 1, 2)
            chi = hf.get('chi')
            chi = torch.tensor(chi).permute(-1, 0, 1, 2)
            sample = {'phase': phase, 'chi': chi}
            if self.transform:
                sample = self.transform(sample)
            return sample

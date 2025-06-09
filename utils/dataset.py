import os
import torch
from torch.utils.data import Dataset


class GeometricFiguresDataset(Dataset):
    """
    Dataset of random geometric figures composed by cylinders and spheres in different scan angles
    """
    def __init__(self, phase_dir, chi_dir, transform=None):
        self.phase_dir = phase_dir
        self.chi_dir = chi_dir
        self.transform = transform
        self.phase_images = os.listdir(self.phase_dir)
        self.chi_images = os.listdir(self.chi_dir)

    def __len__(self):
        return len(self.chi_images)

    def __getitem__(self, item):
        phase_name = os.path.join(self.phase_dir, self.phase_images[item])
        chi_name = os.path.join(self.chi_dir, self.chi_images[item])
        phase = torch.load(phase_name).permute(-1, 0, 1, 2)
        chi = torch.load(chi_name).permute(-1, 0, 1, 2)
        sample = {'phase': phase, 'chi': chi}
        if self.transform:
            sample = self.transform(sample)
        return sample

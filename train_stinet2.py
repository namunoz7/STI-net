#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Nestor Muñoz, 08/05/2025 13:06
Reconstruct the susceptibility tensor for only one direction.

"""

import torch
import h5py
import numpy as np
import os
import torch.nn as nn
from torch.optim import lr_scheduler
from Models.Pytorch.STIResNetSplit_angles import STIResNetSplit
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_total_phase, add_gauss_noise, image_grid, check_root, projection_variables, load_dir_field

# Tensorboard parameters
train_writer = SummaryWriter()
val_writer = SummaryWriter()

# Dataset parameters
CHI_FOLDER = '../Dataset/Chi/'
MAT_FOLDER = '../Dataset/Matrix_Projection/'
DIRECTION_FILE = '../Phantom_real_data/Susceptibility_data/direction_field.mat'

# STINet parameters
IN_CHANNELS = 6
OUT_CHANNELS = (3, 3)
FEATURES = 32
STRIDE = 2
CHI_SIZE = (64, 64, 64, 6)
PHI_SIZE = (64, 64, 64, 6)
MATRIX_SIZE = (64, 64, 64, 6, 6)
SPACE_SIZE = torch.tensor([64, 64, 64])
RESOLUTION = torch.tensor([0.1, 0.1, 0.1])
FOV = SPACE_SIZE * RESOLUTION

# Training parameters
NUM_EPOCHS = 300
BATCH_SIZE = 20
NUM_CHI = 40000
N_ORIENTATIONS = 6
REG_LAMBDA = 6.25e-5
LR = 8e-4
BETAS = (0.5, 0.99)
ANI_REG = 3
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
STD_NOISE = 1e-3
DEVICE = torch.device('cuda')
FINE_TUNING = False
CHECKPOINT_FOLDER = 'checkpoints/Pytorch/STInet_res-unet2/'
CHECKPOINT_NAME = os.path.join(CHECKPOINT_FOLDER, 'state_dicts_sti.pt')


# Models
sti_net = STIResNetSplit(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
                         init_features=FEATURES, stride=STRIDE).to(DEVICE)

# Optimizers
optimizer_net = torch.optim.Adam(params=sti_net.parameters(), lr=LR, betas=BETAS)

# Losses
l1_loss = nn.L1Loss(reduction='mean').to(DEVICE)
l2_loss = nn.MSELoss(reduction='mean').to(DEVICE)

# Schedulers
lr_scheduler_gen = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_net, T_max=100, eta_min=5e-4)


def identity_loss(real_image, same_image, reg_parameter):
    """
    L2 loss from the difference between the generated STI and the dataset STI
    :param real_image: Image from the data
    :param same_image: Generator that is used to make the same image
    :param reg_parameter: Regularization parameter
    :return: Identity Loss
    """
    loss = l2_loss(real_image, same_image)
    return reg_parameter * loss


def cycle_loss(real_image, recovered_image, weight):
    """
    Generates the supervised loss of the tensor
    :param real_image: Image from the data
    :param recovered_image: Cycled image, passed from the two generators
    :param weight: Regularization parameter
    :return: Cycle loss
    """
    loss = l1_loss(real_image, recovered_image)
    return weight * loss


def training_step(chi, matrix_projection, direction_field):
    """
    Training step of the generator using Pytorch
    :param direction_field:
    :param chi: Susceptibility tensor
    :param matrix_projection: Matrix of the linear model of the susceptibility tensor
    :return: train_loss
    """
    iso_chi = torch.stack([chi[:, :, :, :, 0], chi[:, :, :, :, 3], chi[:, :, :, :, 5]], dim=4)
    ani_chi = torch.stack([chi[:, :, :, :, 1], chi[:, :, :, :, 2], chi[:, :, :, :, 4]], dim=4)
    phi = get_total_phase(chi, matrix_projection)
    phi_in = add_gauss_noise(phi, STD_NOISE)
    optimizer_net.zero_grad()
    with torch.set_grad_enabled(True):
        chi_model = sti_net(phi_in, direction_field)
        phi_model = get_total_phase(chi_model, matrix_projection)
        iso_model = torch.stack([chi_model[:, :, :, :, 0], chi_model[:, :, :, :, 3],
                                 chi_model[:, :, :, :, 5]], dim=4)
        ani_model = torch.stack([chi_model[:, :, :, :, 1], chi_model[:, :, :, :, 2],
                                 chi_model[:, :, :, :, 4]], dim=4)
        iso_loss = cycle_loss(real_image=iso_chi, recovered_image=iso_model, weight=1)
        ani_loss = cycle_loss(real_image=ani_chi, recovered_image=ani_model, weight=ANI_REG)
        reg_loss = identity_loss(phi, phi_model, REG_LAMBDA)
        loss = iso_loss + ani_loss + reg_loss
        loss.backward()
        optimizer_net.step()
    return [iso_loss, ani_loss, loss]


def val_step(chi, matrix_projection, direction_field):
    """
    Performs a validation step of the sti model
    :param direction_field:
    :param chi: Susceptibility images
    :param matrix_projection:
    :return:
    """
    iso_chi = torch.stack([chi[:, :, :, :, 0], chi[:, :, :, :, 3], chi[:, :, :, :, 5]], dim=4)
    ani_chi = torch.stack([chi[:, :, :, :, 1], chi[:, :, :, :, 2], chi[:, :, :, :, 4]], dim=4)
    phi = get_total_phase(chi, matrix_projection)
    phi = add_gauss_noise(phi, STD_NOISE)
    with torch.set_grad_enabled(False):
        chi_model = sti_net(phi, direction_field)
        iso_model = torch.stack([chi_model[:, :, :, :, 0], chi_model[:, :, :, :, 3],
                                 chi_model[:, :, :, :, 5]], dim=4)
        ani_model = torch.stack([chi_model[:, :, :, :, 1], chi_model[:, :, :, :, 2],
                                 chi_model[:, :, :, :, 4]], dim=4)
        iso_loss = cycle_loss(real_image=iso_chi, recovered_image=iso_model, weight=1)
        ani_loss = cycle_loss(real_image=ani_chi, recovered_image=ani_model, weight=ANI_REG)
        loss = iso_loss + ani_loss
    return [iso_loss, ani_loss, loss]


def main():
    class GeometricFiguresDataset(Dataset):
        """
        Dataset of random geometric figures composed by cylinders and spheres
        """

        def __init__(self, dataset_dir, key_dataset, num_dataset, transform=None):
            self.dataset_dir = dataset_dir
            self.key = key_dataset
            self.transform = transform
            self.num_dataset = num_dataset
            self.data_images = ['data_' + x + '.h5' for x in [str(y) for y in [*range(num_dataset)]]]

        def __len__(self):
            return self.num_dataset

        def __getitem__(self, item):
            filename = os.path.join(self.dataset_dir, self.data_images[item])
            with h5py.File(filename, 'r') as hf:
                out_tensor = torch.tensor(np.array(hf.get(self.key))).to(torch.float32)
                return out_tensor

    # Dataset
    chi_dataset = GeometricFiguresDataset(dataset_dir=CHI_FOLDER, key_dataset='chi', num_dataset=NUM_CHI)
    n_train = int(len(chi_dataset) * TRAIN_SPLIT)
    n_val = int(len(chi_dataset) * VAL_SPLIT)
    train_dataset, val_dataset = random_split(chi_dataset, [n_train, n_val],
                                              generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=1)

    can_do_ft = check_root(CHECKPOINT_FOLDER)
    if can_do_ft and FINE_TUNING:
        print('Fine tuning')
        checkpoint = torch.load(CHECKPOINT_NAME)
        sti_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer_net.load_state_dict(checkpoint['optimizer_generator_state_dict'])
        lr_scheduler_gen.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_loss']
        train_loss = checkpoint['train_gen_loss']
        train_loss_iso = checkpoint['train_loss_iso']
        train_loss_ani = checkpoint['train_loss_ani']
        val_loss = checkpoint['val_loss']
        val_loss_iso = checkpoint['val_loss_iso']
        val_loss_ani = checkpoint['val_loss_ani']
    else:
        print('Loading parameters')
        # Initial parameters
        best_loss = 1000.0
        train_loss = np.empty(NUM_EPOCHS)
        train_loss_iso = np.empty(NUM_EPOCHS)
        train_loss_ani = np.empty(NUM_EPOCHS)
        val_loss = np.empty(NUM_EPOCHS)
        val_loss_iso = np.empty(NUM_EPOCHS)
        val_loss_ani = np.empty(NUM_EPOCHS)
    print('Starting training')
    print('--------------------------')
    direction_field = load_dir_field(DIRECTION_FILE).view(18).unsqueeze(0).repeat(BATCH_SIZE, 1).to(DEVICE)
    mat_projection = projection_variables(dir_field_file=DIRECTION_FILE, mat_size=SPACE_SIZE, fov=FOV).to(DEVICE)
    for n_epoch in range(NUM_EPOCHS):
        print('------------------------------------')
        print('Epoch number:' + str(n_epoch + 1))
        n_img = np.random.randint(0, BATCH_SIZE)
        # TRAINING PHASE
        print('Training phase')
        sti_net.train()
        actual_loss_gen = 0.0
        iso_train_loss = 0.0
        ani_train_loss = 0.0
        tmp = 0
        for n_chi, chi in enumerate(train_loader):
            if n_chi*BATCH_SIZE % 200 == 0:
                print('Training iteration: ' + str(n_chi))
            tmp += 1
            chi = chi.to(DEVICE, non_blocking=True)
            [iso_loss, ani_loss, gen_loss] = training_step(chi, matrix_projection=mat_projection,
                                                           direction_field=direction_field)
            actual_loss_gen += gen_loss.item()
            iso_train_loss += iso_loss.item()
            ani_train_loss += ani_loss.item()

        train_loss[n_epoch] = actual_loss_gen / tmp
        train_loss_iso[n_epoch] = iso_train_loss / tmp
        train_loss_ani[n_epoch] = ani_train_loss / tmp

        # Tensorboard Loss curves
        train_writer.add_scalars(main_tag='Train_Loss',
                                 tag_scalar_dict={'Global Loss': train_loss[n_epoch],
                                                  'Anisotropic_loss': train_loss_ani[n_epoch],
                                                  'Isotropic_loss': train_loss_iso[n_epoch]},
                                 global_step=n_epoch)
        train_writer.add_scalar(tag='Train_Loss', scalar_value=train_loss[n_epoch],
                                global_step=n_epoch)
        train_writer.close()
        train_writer.flush()

        # VALIDATION PHASE
        actual_val_loss = 0.0
        iso_val_loss = 0.0
        ani_val_loss = 0.0
        tmp = 0
        print('Validation phase')
        sti_net.eval()
        for n_chi, chi in enumerate(val_loader):
            if n_chi % 200 == 0:
                print('Validation iteration: ' + str(n_chi))
            tmp += 1
            chi = chi.to(DEVICE, non_blocking=True)
            [iso_gen_loss, ani_gen_loss, gen_val_loss] = val_step(chi, mat_projection, direction_field)
            actual_val_loss += gen_val_loss.item()
            iso_val_loss += iso_gen_loss.item()
            ani_val_loss += ani_gen_loss.item()
            # del chi
            # torch.cuda.empty_cache()
        val_loss[n_epoch] = actual_val_loss / tmp
        val_loss_iso[n_epoch] = iso_val_loss / tmp
        val_loss_ani[n_epoch] = ani_val_loss / tmp

        chi_data = next(iter(val_loader)).to(DEVICE)
        phase_data = get_total_phase(chi_data, mat_projection)
        chi_model = sti_net(phase_data, direction_field)
        fig1 = image_grid(tensor=chi_data, img=n_img, chi_scale=[-0.15, 0.15])
        fig2 = image_grid(tensor=chi_model, img=n_img, chi_scale=[-0.15, 0.15])
        fig3 = image_grid(tensor=(chi_data - chi_model), img=n_img, chi_scale=[-0.05, 0.05])

        # Tensorboard Loss curves and Visualization images
        val_writer.add_scalar(tag='Validation_Loss', scalar_value=val_loss[n_epoch], global_step=n_epoch)
        val_writer.add_scalars(main_tag='Validation_Loss',
                               tag_scalar_dict={'Anisotropic_loss': val_loss_ani[n_epoch],
                                                'Isotropic_loss': val_loss_iso[n_epoch]},
                               global_step=n_epoch)
        val_writer.add_figure(tag='Ground truth', figure=fig1, global_step=n_epoch)
        val_writer.add_figure(tag='Model output', figure=fig2, global_step=n_epoch)
        val_writer.add_figure(tag='Ground truth - Model output', figure=fig3, global_step=n_epoch)
        val_writer.close()
        val_writer.flush()

        if actual_val_loss < best_loss:
            best_loss = actual_val_loss
            torch.save({
                'n_epoch': n_epoch,
                'model_state_dict': sti_net.state_dict(),
                'optimizer_generator_state_dict': optimizer_net.state_dict(),
                'train_gen_loss': train_loss,
                'val_loss': val_loss,
                'train_loss_iso': train_loss_iso,
                'train_loss_ani': train_loss_ani,
                'val_loss_iso': val_loss_iso,
                'val_loss_ani': val_loss_ani,
                'best_loss': best_loss}, CHECKPOINT_NAME)


if __name__ == '__main__':
    main()

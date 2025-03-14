#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:47:45 2019

@author: Nestor Munoz
"""

import numpy as np
import torch
import h5py
import rand_parameters as rp
import sti_functions as sti

MAT_SIZE = np.array([72, 72, 72])
N_ORIENTATIONS = 6
N_SET_ORIENTATIONS = 11
START = 0
NUM_FIGURES = 1
RESOLUTION = np.array([0.1, 0.1, 0.1])
FOV = MAT_SIZE * RESOLUTION
MAT_SIZE = tuple(MAT_SIZE)
WINDOW = 64
DELTA_FOV = ((np.array(MAT_SIZE) - WINDOW) // 2)
MAX_ANGLE = 40.0
# RESULTS_FOLDER = '/mnt/researchers/cristian-tejos/datasets/STI_dataset/Matrix_Projection/'
# RESULTS_FOLDER = '../../Dataset/Matrix_Projection/'
RESULTS_FOLDER = '../../Imagenes/Dataset/Matrix_Projection/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_direction_field(dir_field):
    """
    Gets the angle of the acquired set of images
    :param dir_field: complete
    :return:
    """
    return dir_field.view(-1)


def main():
    for n_img in range(START, NUM_FIGURES):
        file = 'data_' + str(n_img) + '.h5'
        file_name = RESULTS_FOLDER + file

        vec_theta, vec_psi = rp.angles_cylinders(N_ORIENTATIONS, MAX_ANGLE, N_SET_ORIENTATIONS)
        matrix_projection, direction_field = sti.projection_variables(vec_theta, vec_psi, N_ORIENTATIONS)
        direction_field = process_direction_field(direction_field)
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset(name='A', data=matrix_projection[DELTA_FOV[0] - 1:MAT_SIZE[0] - DELTA_FOV[0] - 1,
                              DELTA_FOV[1] - 1:MAT_SIZE[1] - DELTA_FOV[1] - 1,
                              DELTA_FOV[2] - 1:MAT_SIZE[2] - DELTA_FOV[2] - 1, :, :])
            hf.create_dataset(name='H', data=direction_field)
        print(file + ' saved')


if __name__ == '__main__':
    main()

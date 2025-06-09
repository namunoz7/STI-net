import torch
import h5py
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import sys
import os
import io
import scipy

DEVICE = torch.device('cuda')
DTYPE = torch.float32


def show_img(fig, img, msg, n_rows=1, n_cols=1, ii=1):
    """
    'Saves' a grid of images. I use it to show the susceptibility tensor as in a matrix (Tensor)
    :param fig:
    :param img: Slice of the image to show. Must be of rank 2
    :param msg: Title to show
    :param n_rows:
    :param n_cols:
    :param ii:
    :return:
    """
    print(msg)
    axis = fig.add_subplot(n_rows, n_cols, ii)
    axis.set_title(msg)
    axis.imshow(img, cmap='gray')


def imshow_phi(tensor, img, scale=None, legends=None):
    """
    Returns a 6x1 grid susceptibility images, displaying the susceptibility tensor as a matplotlib figure
    :param scale: Range to plot the images
    :param legends: Names of the figures in the subplot
    :param img: Slice of the images to plot
    :param tensor: Susceptibility image tensor to be displayed
    :return: figure
    """
    # Create a figure containing the plot
    if scale is None:
        scale = [-0.1, 0, 1]
    figure = plt.figure()
    tmp_1 = [1, 2, 3, 5, 6, 7]
    for n in range(0, 6):
        # Start next subplot
        figure.add_subplot(6, 1, tmp_1[n])
        plt.xticks([])
        plt.yticks([])
        plt.tick_params(color='gray')
        if legends is not None:
            plt.text(20, 40, legends[n], color='white', fontsize=18)
        plt.imshow(np.fliplr(np.rot90(tensor[:, :, img, n])), cmap='gray', vmin=scale[0], vmax=scale[1])
    plt.subplots_adjust(wspace=0, hspace=0)
    return figure


def read_h5_image(filename, key):
    """
    Read a h5 file image that contains the bulk phase and the susceptibility tensor of few geometric figures
    :param filename: Name of the file to read
    :param key:
    :return: chi, phase
    """
    with h5py.File(filename, 'r') as hf:
        out_tensor = torch.tensor(np.array(hf.get(key)))
        out_tensor = out_tensor.to(torch.float32)
        return out_tensor


def image_grid(tensor, img, chi_scale):
    """
    Returns a 3x3 grid susceptibility images, displaying the susceptibility tensor as a matplotlib figure
    :param chi_scale:
    :param img: number of batched image to plot
    :param tensor: Susceptibility image tensor to be displayed
    :return: figure
    """
    # Create a figure containing the plot
    tensor2 = tensor.detach().cpu()
    figure = plt.figure()
    indexes = [1, 2, 3, 5, 6, 9]
    titles = [r'$\chi_{11}$', r'$\chi_{12}$', r'$\chi_{13}$', r'$\chi_{22}$', r'$\chi_{23}$', r'$\chi_{33}$']
    for n in range(0, 6):
        # Start next subplot
        plt.subplot(3, 3, indexes[n], title=(titles[n]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(tensor2[img, :, :, 24, n], cmap='gray', vmin=chi_scale[0], vmax=chi_scale[1])
    cax = plt.axes((0.05, 0.07, 0.6, 0.035))
    plt.colorbar(cax=cax, orientation='horizontal')
    plt.subplots_adjust(left=0, right=0.01, bottom=0, top=0.5, wspace=0)
    figure.tight_layout()
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to an PNG image, and returns it. The supplied figure is closed
    and inaccessible after this call
    :param figure: Matplotlib figure to plot in tensorboard
    :return: image
    """
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figures prevents it from being displayed
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to tf image
    image = np.frombuffer(buf, dtype=np.uint8)
    return image


def echo_line(phase_idx, epoch, batch_idx, batch_size, n_dataset, actual_loss):
    """
    Prints the line of the run, with the loss and the percentage of data run
    """
    if batch_idx % 20 == 0:
        print(phase_idx + ' Epoch: {} [{}/{}({:.0f}%)] \tLoss: {:.4e}'.format(
            epoch + 1, (batch_idx + 1) * batch_size, n_dataset,
            100 * (batch_idx + 1) * batch_size / n_dataset, actual_loss))


def echo_line_2(phase_idx, epoch, n_trained, n_dataset, actual_loss):
    """
    Prints the line of the run, with the loss and the percentage of data run
    """
    # if batch_idx % 20 == 0:
    print(phase_idx + ' Epoch: {} [{}/{}({:.0f}%)] \tLoss: {:.4e}'.format(
        epoch + 1, n_trained, n_dataset,
        100 * n_trained / n_dataset, actual_loss))


def total_variation(img):
    tv_d = torch.abs(img[:, :, 1:, :, :] - img[:, :, :-1, :, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:, :] - img[:, :, :, :-1, :]).sum()
    tv_h = torch.abs(img[:, :, :, :, 1:] - img[:, :, :, :, :-1]).sum()
    return tv_d + tv_w + tv_h


def read_img(filename):
    """
    Read nifty image
    :param filename:
    :return:
    """
    nii_img = nib.load(filename)
    img = nii_img.get_fdata()
    img = torch.from_numpy(img)
    img_shape = np.array(img.shape[0:3])
    header = nii_img.header
    voxel_size = np.array(header.get_zooms()[0:3])
    fov = img_shape * voxel_size

    return img, img_shape, fov, nii_img


def fft_phase(inputs):
    """
    Gets the fourier transform of the input geometric figure images.
    :param inputs: Input image
    :return:
    """
    fft_input = torch.fft.fftn(input=inputs, dim=(0, 1, 2))
    fft_input = torch.fft.fftshift(fft_input, dim=(0, 1, 2, 3))
    return fft_input


def inv_fft_phase(fft_input):
    """
    Gets the inverse Fourier Transform
    :param fft_input:
    :return:
    """
    fft_input = torch.fft.ifftshift(fft_input, dim=(0, 1, 2))
    inputs = torch.fft.ifftn(fft_input, dim=(0, 1, 2))
    return inputs


def get_total_phase(chi, matrix_projection):
    """
    Calculates the total phase of the tensor using the linear form of the Tensor model
    :param matrix_projection:
    :param chi: Vectorized form of the tensor
    :return: phase: the local phase of the image
    """
    fft_chi = fft_phase(chi)
    tmp_real = torch.matmul(matrix_projection, torch.real(fft_chi).unsqueeze(-1))
    tmp_img = torch.matmul(matrix_projection, torch.imag(fft_chi).unsqueeze(-1))
    tmp_phi = torch.cat((tmp_real, tmp_img), dim=-1)
    phi = inv_fft_phase(torch.view_as_complex(tmp_phi)).to(DEVICE)
    return torch.real(phi)


def add_gauss_noise(phi, std_noise):
    """
    Generates noise in the
    :param phi: phase image of the tensor
    :param std_noise: Standard deviation of the gaussian noise
    :return:
    """
    phi += std_noise * torch.rand(phi.size()).to(DEVICE)
    return phi


def shift_fft(x, dims, mat_size):
    """
    Shift zero-frequency component to center of spectrum
    :param mat_size:
    :param x: Input image
    :param dims: Dimensions to roll
    :return:
    """
    x = x.roll((torch.div(mat_size[0], 2, rounding_mode='trunc'),
                torch.div(mat_size[1], 2, rounding_mode='trunc'),
                torch.div(mat_size[2], 2, rounding_mode='trunc')), dims)
    return x


def get_eigenvectors(tensor):
    """
    Generates an eigen-decomposition and calculates the eigenvectors of the tensor
    :param tensor: Input tensor
    :return: eigenvectors
    """
    tensor = torch.stack([tensor[:, :, :, :, 0], tensor[:, :, :, :, 1], tensor[:, :, :, :, 2],
                          tensor[:, :, :, :, 1], tensor[:, :, :, :, 3], tensor[:, :, :, :, 4],
                          tensor[:, :, :, :, 2], tensor[:, :, :, :, 4], tensor[:, :, :, :, 5]], dim=-1)
    mat_size = tensor.size()
    tensor = torch.reshape(tensor, shape=(mat_size[0], mat_size[1], mat_size[2], mat_size[3], 3, 3))
    _, eigenvectors = torch.linalg.eigh(tensor, UPLO='U')
    return torch.unsqueeze(eigenvectors[:, :, :, :, 0], dim=-1)


def get_angle_images(pev_gt, pev_model, reg_weight):
    """
    Calculates the cosine of both principal eigenvectors.
    :param reg_weight:
    :param pev_gt:
    :param pev_model:
    :return:
    """
    pev_model = torch.unsqueeze(pev_model, dim=-2)
    dot_prod = torch.matmul(pev_model, pev_gt)
    dot_prod = -reg_weight * torch.mean(torch.abs(torch.squeeze(dot_prod)))
    return torch.squeeze(dot_prod)


def angles_cylinders(num_rotations, theta, psi):
    """
    Rotation angles that are used to rotate the object in the scanner
    :param num_rotations: Number of rotations that will be scanned
    :param theta: Angle of rotation in the y-axis (LR axis)
    :param psi: Angle of rotation in the x-axis (AP axis)
    """
    min_rand = -5.0
    max_rand = 5.0

    tmp = (max_rand - min_rand) * torch.rand((1, num_rotations), dtype=torch.float64) + min_rand
    # Tilt angle of main field
    if num_rotations == 6:
        vec_theta = torch.tensor([0.0, 0.0, 0.0, theta/2, theta/2, theta], dtype=torch.float64)
    else:
        vec_theta = torch.tensor([0.0, 0.0, 0.0, theta/2, theta/2, theta/2, theta/2, theta, theta, theta, theta, theta],
                                 dtype=torch.float64)
    vec_theta = vec_theta.reshape(1, num_rotations)
    vec_theta = vec_theta + tmp
    vec_theta = torch.deg2rad(vec_theta)

    tmp = (max_rand - min_rand) * torch.rand((1, num_rotations), dtype=torch.float64) + min_rand
    # Rotation angle with the z axis
    if num_rotations == 6:
        vec_psi = torch.tensor([0.0, -psi, psi, -psi/2, psi/2, 0.0], dtype=torch.float64)
    else:
        vec_psi = torch.tensor([0.0, -psi, psi, -psi/2, psi/2, -psi, psi, -psi/2, psi/2, -psi, psi, 0.0],
                               dtype=torch.float64)
    vec_psi = vec_psi.reshape(1, num_rotations)
    vec_psi += tmp
    vec_psi = torch.deg2rad(vec_psi)

    return vec_theta, vec_psi


def get_direction_field(vec_theta, vec_psi, n_orientations):
    """
    Gets the direction field vector of the multiple orientations, made by the cylinders. All the angles are in radians
    :param vec_theta: Rotation angle in the x-z plane
    :param vec_psi: Rotation angle in the y-z plane
    :param n_orientations:
    :return:
    """
    direction_field = torch.zeros(n_orientations, 3, dtype=torch.float64)
    direction_field[:, 0] = torch.sin(vec_theta)  # Hx
    direction_field[:, 1] = -torch.sin(vec_psi) * torch.cos(vec_theta)  # Hy
    direction_field[:, 2] = torch.cos(vec_psi) * torch.cos(vec_theta)  # Hz
    direction_field = direction_field.unsqueeze(-1)

    return direction_field


def gen_k_space(fov, n_orientations, mat_size):
    """
    Defines the K space
    :param mat_size:
    :param fov:
    :param n_orientations:
    :return:
    """
    kx = torch.arange(1, mat_size[0] + 1, dtype=torch.float32)
    ky = torch.arange(1, mat_size[1] + 1, dtype=torch.float32)
    kz = torch.arange(1, mat_size[2] + 1, dtype=torch.float32)

    center_x = torch.div(mat_size[0], 2, rounding_mode='trunc') + 1
    center_y = torch.div(mat_size[1], 2, rounding_mode='trunc') + 1
    center_z = torch.div(mat_size[2], 2, rounding_mode='trunc') + 1
    kx = kx - center_x
    ky = ky - center_y
    kz = kz - center_z

    delta_kx = 1 / fov[0]
    delta_ky = 1 / fov[1]
    delta_kz = 1 / fov[2]

    #  Generation of k space

    kx = kx * delta_kx
    ky = ky * delta_ky
    kz = kz * delta_kz

    kxx, kyy, kzz = torch.meshgrid(kx, ky, kz)

    kxx = kxx.unsqueeze(3).repeat(1, 1, 1, n_orientations)
    kyy = kyy.unsqueeze(3).repeat(1, 1, 1, n_orientations)
    kzz = kzz.unsqueeze(3).repeat(1, 1, 1, n_orientations)

    k = torch.zeros(mat_size[0], mat_size[1], mat_size[2], n_orientations, 3, dtype=torch.float32)
    k[:, :, :, :, 0] = kxx
    k[:, :, :, :, 1] = kyy
    k[:, :, :, :, 2] = kzz
    k = k.unsqueeze(-1)

    return k, kxx, kyy, kzz


def projection_variables(dir_field_file, fov, mat_size):
    """
    Generates the projection variables of the STI model (a_ii and a_ij) and construct the projection matrix
    These are only made with 12 different orientation
    :param mat_size:
    :param fov:
    :param dir_field_file: File name of the direction field.
    :return: A: matrix containing each one of the projection angles.
    """
    print("Generating projection variables...")
    n_orientations = 6
    eps = sys.float_info.epsilon
    k, kxx, kyy, kzz = gen_k_space(fov, n_orientations, mat_size)
    k2 = (kxx * kxx) + (kyy * kyy) + (kzz * kzz)
    k2[k2 == 0] = eps
    direction_field = load_dir_field(dir_field_file)
    kt_h = torch.matmul(k.transpose(-2, -1), direction_field).squeeze()
    direction_field = direction_field.squeeze()
    # Aux variables are defined
    a_11 = ((direction_field[:, 0] * direction_field[:, 0]) / 3) - ((kt_h / k2) * kxx * direction_field[:, 0])
    a_22 = ((direction_field[:, 1] * direction_field[:, 1]) / 3) - ((kt_h / k2) * kyy * direction_field[:, 1])
    a_33 = ((direction_field[:, 2] * direction_field[:, 2]) / 3) - ((kt_h / k2) * kzz * direction_field[:, 2])
    a_12 = ((2 / 3) * direction_field[:, 0] * direction_field[:, 1]) - \
           ((kt_h / k2) * (kxx * direction_field[:, 1] + kyy * direction_field[:, 0]))
    a_13 = ((2 / 3) * direction_field[:, 0] * direction_field[:, 2]) - \
           ((kt_h / k2) * (kxx * direction_field[:, 2] + kzz * direction_field[:, 0]))
    a_23 = ((2 / 3) * direction_field[:, 2] * direction_field[:, 1]) - \
           ((kt_h / k2) * (kzz * direction_field[:, 1] + kyy * direction_field[:, 2]))
    print('...... Done')
    matrix_projection = torch.zeros(mat_size[0], mat_size[1], mat_size[2], n_orientations, 6, dtype=torch.float64)
    matrix_projection[:, :, :, :, 0] = a_11
    matrix_projection[:, :, :, :, 1] = a_12
    matrix_projection[:, :, :, :, 2] = a_13
    matrix_projection[:, :, :, :, 3] = a_22
    matrix_projection[:, :, :, :, 4] = a_23
    matrix_projection[:, :, :, :, 5] = a_33

    matrix_projection = shift_fft(matrix_projection, (0, 1, 2), mat_size)
    matrix_projection = matrix_projection.to(torch.float32)

    return matrix_projection


def get_phase(chi, matrix_projection, std_noise):
    """

    :param chi:
    :param matrix_projection:
    :param std_noise:
    :return:
    """
    def gauss_noise(local_field, noise):
        """
        Generates noise in the
        :param local_field: phase image of the tensor
        :param noise: Standard deviation of the gaussian noise
        :return:
        """
        local_field += noise * torch.rand(local_field.size())
        return local_field

    fft_chi = fft_phase(chi)
    tmp_real = torch.matmul(matrix_projection, torch.real(fft_chi).unsqueeze(-1))
    tmp_img = torch.matmul(matrix_projection, torch.imag(fft_chi).unsqueeze(-1))
    tmp_phi = torch.cat((tmp_real, tmp_img), dim=-1)
    phi = inv_fft_phase(torch.view_as_complex(tmp_phi))
    phi = gauss_noise(phi.real, std_noise)
    return torch.real(phi)


def min_squares(bulk_phase, mat_projection):
    """
    Calculates the tensor by getting the phase of geometric figures.
    :param mat_projection:
    :param bulk_phase: Total phase of the image phase
    :return: The vectorized form of the tensor
    """
    mat_transpose = mat_projection.transpose(3, 4)
    b_mat = torch.matmul(mat_transpose, mat_projection)
    b_inv = b_mat.inverse()

    fft_phi = fft_phase(bulk_phase)
    tmp_real = torch.matmul(mat_transpose, torch.real(fft_phi).unsqueeze(-1))
    tmp_img = torch.matmul(mat_transpose, torch.imag(fft_phi).unsqueeze(-1))
    real_ft_chi = torch.matmul(b_inv, tmp_real).unsqueeze(-1)
    img_ft_chi = torch.matmul(b_inv, tmp_img).unsqueeze(-1)
    tmp_chi = torch.cat((real_ft_chi, img_ft_chi), dim=-1)
    chi = inv_fft_phase(torch.view_as_complex(tmp_chi))
    return torch.real(chi)


def save_img(nii, img, name):
    """
    Save the image as a nifti file
    :param nii: Nifti variable to save
    :param img: Tensor to save
    :param name: Name of the tensor
    """
    hdr = nii.header
    new_dim = img.shape
    hdr.set_data_shape(new_dim)
    nii = nib.Nifti1Image(img, affine=None, header=hdr)
    nib.save(nii, name)
    print('...... ' + name + ' saved')


def mask_image(img, mask):
    """
    Mask the input image with the mask, to obtain only the tissue inside the brain.
    :param img:
    :param mask:
    :return:
    """
    n_orientations = img.shape[-1]
    img = torch.mul(img, mask.unsqueeze(-1).repeat([1, 1, 1, n_orientations]))
    return img


def get_eigen_decomposition(tensor):
    """
    Generates the eigen decomposition of the STI tensor
    :param tensor: Susceptibility tensor in vector form
    :return: eigen_values and eigen_vectors
    """
    mat_size = tensor.size()
    tensor = torch.stack(tensors=[tensor[:, :, :, 0], tensor[:, :, :, 1], tensor[:, :, :, 2],
                                  tensor[:, :, :, 1], tensor[:, :, :, 3], tensor[:, :, :, 4],
                                  tensor[:, :, :, 2], tensor[:, :, :, 4], tensor[:, :, :, 5]], dim=-1)
    tensor = torch.reshape(tensor, shape=(mat_size[0], mat_size[1], mat_size[2], 3, 3))
    eig_val, eig_vec = torch.linalg.eigh(tensor)
    eig_val = torch.stack([eig_val[:, :, :, 2], eig_val[:, :, :, 1], eig_val[:, :, :, 0]], dim=-1)
    v1 = eig_vec[:, :, :, :, 0]
    return eig_val, v1


def gen_mms_msa(tensor):
    """
    Calculates the mean magnetic susceptibility (MMS) and magnetic susceptibility anisotropy (MSA)
    :param tensor: Susceptibility tensor input in vectorized form
    :return: mms, msa
    """
    eigenvalues, v1 = get_eigen_decomposition(tensor)
    mms = torch.mean(eigenvalues, dim=-1)
    msa = eigenvalues[:, :, :, 0] - torch.mean(eigenvalues[:, :, :, 1:3], dim=-1)
    return v1, mms, msa


def get_angle_images_test(pev_1, pev_2):
    """
    Calculates the cosine of both principal eigenvectors.
    :param pev_1:
    :param pev_2:
    :return:
    """
    pev_1 = torch.unsqueeze(pev_1, dim=-1)
    pev_2 = torch.unsqueeze(pev_2, dim=-2)
    dot_prod = torch.matmul(pev_2, pev_1)
    dot_prod = torch.abs(torch.squeeze(dot_prod))
    return torch.squeeze(dot_prod)


def load_dir_field(dir_field_file, directions=(0, 1, 2, 3, 4, 5)):
    """
    Load the direction field at which the scanning process was performed
    :param dir_field_file:
    :param directions:
    :return:
    """
    dir_field = scipy.io.loadmat(dir_field_file)
    dir_field = dir_field['direction_field']
    dir_field = torch.tensor(dir_field[directions, :], dtype=torch.float32).unsqueeze(-1)
    return dir_field


def check_root(root_file):
    """
    Check if the root file exists. If it does not, it creates the root file
    :param root_file:
    :return:
    """
    if not os.path.exists(root_file):
        os.mkdir(root_file)
        print("Directory " + root_file + " Created ")
        return False
    else:
        print("Directory " + root_file + " already exists")
        return True

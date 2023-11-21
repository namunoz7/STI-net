import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def define_space(num_figures, spatial_res, mat_size):
    """
    Create the tensors necessary to simulate the cylinders and spheres
    :param mat_size: Matrix size of the space
    :param spatial_res: Spatial resolution of the image
    :param num_figures: number of figures to simulate
    :return: xx, yy, zz: the space
    """

    x = torch.arange(mat_size[0], dtype=torch.float64, device=DEVICE) * spatial_res[0]
    y = torch.arange(mat_size[0], dtype=torch.float64, device=DEVICE) * spatial_res[1]
    z = torch.arange(mat_size[0], dtype=torch.float64, device=DEVICE) * spatial_res[2]

    xx, yy, zz = torch.meshgrid([x, y, z])

    xx = xx.repeat(num_figures, 1, 1, 1)
    yy = yy.repeat(num_figures, 1, 1, 1)
    zz = zz.repeat(num_figures, 1, 1, 1)

    return xx, yy, zz,


def get_center(num_figures, spatial_res, mat_size):
    """
    Define the center of the space
    :param mat_size: Matrix size of the space
    :param spatial_res:
    :param num_figures: Number of figures to simulate
    :return:
    """
    from generate_dataset import WINDOW
    min_center_x = mat_size[0] // 2 - WINDOW // 2
    min_center_y = mat_size[1] // 2 - WINDOW // 2
    min_center_z = mat_size[2] // 2 - WINDOW // 2
    max_center_x = mat_size[0] // 2 + WINDOW // 2
    max_center_y = mat_size[1] // 2 + WINDOW // 2
    max_center_z = mat_size[2] // 2 + WINDOW // 2
    center_nx = min_center_x + (max_center_x - min_center_x) * torch.rand(num_figures, device=DEVICE)
    center_ny = min_center_y + (max_center_y - min_center_y) * torch.rand(num_figures, device=DEVICE)
    center_nz = min_center_z + (max_center_z - min_center_z) * torch.rand(num_figures, device=DEVICE)

    center_nx = center_nx.round() * spatial_res[0]
    center_ny = center_ny.round() * spatial_res[1]
    center_nz = center_nz.round() * spatial_res[2]

    return center_nx, center_ny, center_nz

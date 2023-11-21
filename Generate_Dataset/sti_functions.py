import torch
import sys
eps = sys.float_info.epsilon
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def crammer_rule(real_sigma, img_sigma, b_mat, det_b, num_chi):
    """
    Takes the chi value of the STI tensor, defined by num_chi
    :param real_sigma:
    :param img_sigma:
    :param b_mat:
    :param det_b:
    :param num_chi:
    :return:
    """

    b_mat[:, :, :, :, num_chi] = real_sigma.squeeze()
    det_real_chi = b_mat.det()
    real_chi = det_real_chi / (det_b + eps)
    b_mat[:, :, :, :, num_chi] = img_sigma.squeeze()
    det_img_chi = b_mat.det()
    img_chi = det_img_chi / (det_b + eps)

    chi_num = torch.stack((real_chi, img_chi), -1)
    return chi_num


def get_susceptibility_values(n_figures, is_cylinder=False):
    """
    Get the susceptibility values of the figures
    :param is_cylinder:
    :param n_figures: Number of figures in the image
    :return:
    """
    if is_cylinder:
        mean_1 = -0.035
        mean_2 = -0.045
        mean_3 = -0.047
        std_1 = 0.0016
        std_2 = 0.0029
        std_3 = 0.0029
        chi_1 = torch.normal(mean=mean_1, std=std_1, size=(n_figures,), dtype=torch.float64, device=DEVICE)
        chi_2 = torch.normal(mean=mean_2, std=std_2, size=(n_figures,), dtype=torch.float64, device=DEVICE)
        chi_3 = torch.normal(mean=mean_3, std=std_3, size=(n_figures,), dtype=torch.float64, device=DEVICE)
        chi = torch.sort(torch.stack((chi_1, chi_2, chi_3), dim=1), dim=1, descending=True)
        chi = torch.diag_embed(chi.values)
    else:
        # Susceptibility inside the spheres
        min_uniform = 0
        max_uniform = 0.08
        chi_1 = (max_uniform - min_uniform) * torch.rand(size=(n_figures,),
                                                         dtype=torch.float64, device=DEVICE) + min_uniform
        chi_2 = (max_uniform - min_uniform) * torch.rand(size=(n_figures,),
                                                         dtype=torch.float64, device=DEVICE) + min_uniform
        chi_3 = (max_uniform - min_uniform) * torch.rand(size=(n_figures,),
                                                         dtype=torch.float64, device=DEVICE) + min_uniform
        chi = torch.sort(torch.stack((chi_1, chi_2, chi_3), dim=1), dim=1, descending=True)
        chi = torch.diag_embed(chi.values)

    return chi


def projection_variables(vec_theta, vec_psi, n_orientations):
    """
    Generates the projection variables of the STI model (a_ii and a_ij) and construct the projection matrix
    These are only made with 12 different orientation
    :param vec_theta: vector containing the angle of deviation with respect to the main field axis
    :param vec_psi: vector containing the angle of rotation of the x-y plane.
    :param n_orientations:
    :return: A: matrix containing each one of the projection angles.
    """
    import K_space_torch as kSpace
    from generate_tensors import FOV

    direction_field = kSpace.get_direction_field(vec_theta, vec_psi, n_orientations)
    k, kxx, kyy, kzz = kSpace.gen_k_space(FOV, n_orientations)

    k2 = (kxx * kxx) + (kyy * kyy) + (kzz * kzz)
    k2[k2 == 0] = eps
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

    matrix_projection = torch.stack([a_11, a_12, a_13, a_22, a_23, a_33], dim=-1)
    return matrix_projection


def get_cylinder_phase(num_cylinders, rho, vec_theta, vec_psi, vec_radius, delta_x):
    """
    Get the off resonance field of the cylinder by the scalar value of each cylinder
    :param num_cylinders: Number of cylinders
    :param rho: Radius of the cylindrical coordinates
    :param vec_theta: Deviation angle with the main magnetic field
    :param vec_psi: Rotation angle un the x-y plane
    :param vec_radius: Vector containing the radius of the
    :param delta_x:
    :return:
    """
    from generate_tensors import MAT_SIZE
    tmp_cylinder = rho < vec_radius
    # Phase inside the cylinder
    cylinder = torch.zeros(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], num_cylinders, 1, dtype=torch.float64)
    cylinder[tmp_cylinder] = 1.0
    delta_x_i = torch.matmul(delta_x, cylinder)
    cos_theta = torch.cos(vec_theta).transpose(0, 1)
    tmp = ((-0.5 * cos_theta * cos_theta) + (1 / 6))
    # phase_in = torch.matmul(delta_x_i, tmp)
    phase_in = delta_x_i * tmp

    # Phase outside the cylinder
    cylinder = torch.ones(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], num_cylinders, 1, dtype=torch.float64)
    cylinder[tmp_cylinder] = 0.0
    rho[rho == 0] = 1.0
    r2_rho2 = (vec_radius * vec_radius) / (rho * rho)
    cylinder = cylinder * r2_rho2.unsqueeze(-1)
    delta_x_o = torch.matmul(delta_x, cylinder)
    cos_2psi = torch.cos(2 * vec_psi).transpose(0, 1)
    sin2_theta = torch.sin(vec_theta).transpose(0, 1)
    phase_out = 0.5 * delta_x_o * cos_2psi * sin2_theta * sin2_theta

    bulk_phase_cylinder = phase_in + phase_out

    return bulk_phase_cylinder


def get_sphere_phase(num_spheres, vec_radius, xx2, yy2, zz2):
    from generate_tensors import MAT_SIZE
    rho_2 = xx2 + yy2 + zz2
    rho = torch.sqrt(rho_2)
    rho_2[rho == 0] = 1
    [chi_i, chi_o] = get_susceptibility_values(num_spheres)
    delta_chi = chi_o - chi_i
    tmp = rho <= vec_radius
    chi_sphere = torch.zeros(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], num_spheres, dtype=torch.float64)
    chi_sphere[tmp] = 1
    chi_sphere = chi_sphere * chi_i

    tmp = rho > vec_radius
    chi_sphere_out = torch.zeros(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], num_spheres, dtype=torch.float64)
    chi_sphere_out[tmp] = 1
    chi_sphere_out = chi_sphere_out * chi_o
    chi_sphere = chi_sphere + chi_sphere_out

    phase = (1 - (2 / 3) * chi_sphere) * ((vec_radius ** 3) * (delta_chi / (3 + delta_chi))) * (
            (2 * zz2 - xx2 - yy2) / (rho_2 ** (5 / 2)))
    phase[rho < vec_radius] = 0
    phase = phase.sum(-1)

    return phase


def diagonal_tensor(num_cylinders):
    """
    Defines the diagonal tensor of the cylinder images
    :param num_cylinders: Number of cylinders in the image
    :return:
    """
    mean_par = -0.025e-6
    std_par = 0.0012e-6
    mean_per = -0.047e-6
    std_per = 0.0032e-6
    chi_par = torch.normal(mean=mean_par, std=std_par, size=(num_cylinders,), dtype=torch.float64)
    chi_per = torch.normal(mean=mean_per, std=std_per, size=(num_cylinders,), dtype=torch.float64)
    chi = torch.eye(3, dtype=torch.float64).repeat(num_cylinders, 1, 1)
    chi[:, 0, 0] = chi_per
    chi[:, 1, 1] = chi_per
    chi[:, 2, 2] = chi_par

    return chi


def eig_vector_tensor(vec_alpha, vec_beta, vec_omega):
    """
    Simulate the direction of the cylinder in the subject frame
    :param vec_omega: Angle of deviation with the x-axis
    :param vec_alpha: Angle of deviation with the z-axis
    :param vec_beta: Angle of deviation with the y-axis
    :return: Matrix containing the eigen vectors of the cylinders.
    """
    sin_alpha = torch.sin(vec_alpha)
    sin_beta = torch.sin(vec_beta)
    sin_omega = torch.cos(vec_omega)
    cos_alpha = torch.cos(vec_alpha)
    cos_beta = torch.cos(vec_beta)
    cos_omega = torch.cos(vec_omega)

    eig_vec = torch.stack([torch.stack([cos_alpha*cos_beta,
                                        sin_alpha*cos_beta*sin_omega - sin_beta*cos_omega,
                                        sin_alpha*cos_beta*cos_omega + sin_beta*sin_omega]),
                           torch.stack([cos_alpha*sin_beta,
                                        sin_alpha*sin_beta*sin_omega + cos_beta*cos_omega,
                                        sin_alpha*sin_beta*cos_omega - cos_beta*sin_omega]),
                           torch.stack([-sin_alpha, cos_alpha*sin_omega, cos_alpha*cos_omega])])\
        .permute(2, 0, 1)

    return eig_vec


def tilt_scan_image(num_figures, mask, mat_rot, is_cylinder):
    """
    Applies the rotation matrix to tilt the object inside the MRI
    :param mat_rot: Rotation matrix of the geometric figures
    :param is_cylinder: If the figure is a cylinder
    :param mask: Mask of the geometric figures
    :param num_figures: Number of figures
    :return: chi: Susceptibility tensor in the subject frame
    """
    from generate_tensors import MAT_SIZE

    chi_i = get_susceptibility_values(num_figures, is_cylinder)
    chi_tensor = torch.matmul(torch.matmul(mat_rot, chi_i), mat_rot.transpose(1, 2)).\
        reshape(num_figures, 9).transpose(0, 1)
    chi_figures = torch.matmul(chi_tensor, mask).squeeze().reshape(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], 3, 3)
    chi = torch.zeros(MAT_SIZE[0], MAT_SIZE[1], MAT_SIZE[2], 6, dtype=torch.float64, device=DEVICE)
    chi[:, :, :, 0] = chi_figures[:, :, :, 0, 0]
    chi[:, :, :, 1] = chi_figures[:, :, :, 0, 1]
    chi[:, :, :, 2] = chi_figures[:, :, :, 0, 2]
    chi[:, :, :, 3] = chi_figures[:, :, :, 1, 1]
    chi[:, :, :, 4] = chi_figures[:, :, :, 1, 2]
    chi[:, :, :, 5] = chi_figures[:, :, :, 2, 2]

    return chi

import torch
import torch.nn as nn
import args_gans

args = args_gans.parser_gans()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
args = args_gans.parser_gans()
cross_entropy = nn.BCELoss().to(device)
batch_size = args.batch_size
ones_batch = torch.ones(batch_size, device=device)
zeros_batch = torch.zeros(batch_size, device=device)
l1_loss = nn.L1Loss(reduction='mean').to(device)


def discriminator_loss(real_output, fake_output):
    """
    Calculates the loss of a discriminator as the cross entropy of the classifiers (discriminators).
    :param real_output: Output if the discriminator by passing the real image (image from the data set)
    :param fake_output: Output of the discriminator by passing the image of the generator
    :return: Complete cross entropy loss (real and fake)
    """
    real_loss = cross_entropy(real_output, ones_batch)
    fake_loss = cross_entropy(fake_output, zeros_batch)
    return (real_loss + fake_loss) * 0.5


def generator_loss(fake_output):
    """
    Binary cross entropy loss obtained from the real value of the input image (0).
    :param fake_output: Classification obtained from the discriminator by having as input the image taken from the
    generator
    :return: Binary cross
    """
    loss = cross_entropy(fake_output, ones_batch)
    return loss


def identity_loss(real_image, gen, weight=10):
    """
    L1 loss obtained by the images from the data and the same image generated
    :param real_image: Image from the data
    :param gen: Generator that is used to make the same image
    :param weight: Regularization parameter
    :return: Identity Loss
    """
    same_image = gen(real_image)
    loss = l1_loss(real_image, same_image)
    return weight * loss


def cycle_loss(real_image, recovered_image, weight=10):
    """
    Cycle loss of the GAN's obtained by the L1 loss
    :param real_image: Image from the data
    :param recovered_image: Cycled image, passed from the two generators
    :param weight: Regularization parameter
    :return: Cycle loss
    """
    loss = l1_loss(real_image, recovered_image)
    return weight * loss

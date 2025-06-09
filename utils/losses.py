import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# l1_loss = tf.keras.losses.MeanAbsoluteError


def discriminator_loss(real_output, fake_output):
    """
    Calculates the loss of a discriminator as the cross entropy of the classifiers (discriminators).
    :param real_output: Output if the discriminator by passing the real image (image from the data set)
    :param fake_output: Output of the discriminator by passing the image of the generator
    :return: Complete cross entropy loss (real and fake)
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) * 0.5


def generator_loss(fake_output):
    """
    Binary cross entropy loss obtained from the real value of the input image (0).
    :param fake_output: Classification obtained from the discriminator by having as input the image taken from the
    generator
    :return: Binary cross
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def identity_loss(real_img, same_img, alpha=0.5):
    """
    L1 loss obtained by the images from the data and the same image generated
    :param real_img: Image from the data
    :param same_img: Generator that is used to make the same image
    :param alpha: Regularization parameter
    :return: Identity Loss
    """
    loss = tf.reduce_mean(tf.abs(real_img, same_img))
    # loss = l1_loss(real_img, same_img)
    return alpha * loss.numpy()


def cycle_loss(real_image, recovered_image, weight=10):
    """
    Cycle loss of the GAN's obtained by the L1 loss
    :param real_image: Image from the data
    :param recovered_image: Cycled image, passed from the two generators
    :param weight: Regularization parameter
    :return: Cycle loss
    """
    loss = tf.reduce_mean(tf.abs(real_image, recovered_image))
    # loss = l1_loss(real_image, recovered_image)
    return weight * loss.numpy()

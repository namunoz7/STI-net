import torch
import tensorflow as tf
from utils.utils import total_variation, echo_line
from utils.parser_gans import parser_gans

args = parser_gans()


@tf.function
def tv_tf(input_tensor):
    """
    Calculates the total variation function of an image to use as a regularization of the optimization problem
    :param input_tensor: Image obtained by the net
    :return: tv_img
    """
    tv_img = tf.reduce_mean(tf.abs(input_tensor[:, 1:, :, :, :] - input_tensor[:, :-1, :, :, :])) + \
        tf.reduce_mean(tf.abs(input_tensor[:, :, 1:, :, :] - input_tensor[:, :, :-1, :, :])) + \
        tf.reduce_mean(tf.abs(input_tensor[:, :, :, 1:, :] - input_tensor[:, :, :, :-1, :]))
    return tv_img


def train_phase(model, loader, mse_loss, optimizer, lambda_reg, epoch, train_loss, device):
    """
    Train phase of the model. It gets loads each input and output batch, calculates the
    output by the model, gets the loss with total variation as regularization and maks an
    optimizer step.
    :param device: GPU to run
    :param model: Model to train
    :param loader: Object to load the images for the train phase
    :param mse_loss: Loss functions
    :param optimizer: Optimizer function for the model parameters
    :param lambda_reg: Regularization parameter for the regularization function
    :param epoch: Actual epoch of the training phase
    :param train_loss: Array to save the loss in each epoch
    :return: model, train_loss
    """
    n_dataset = len(loader['train'].data)
    model.train()
    actual_loss = 0.0
    tmp = 0

    # Iterate over data
    for batch_idx, sample_batch in enumerate(loader['train']):
        tmp += 1
        phase_batch, chi_batch = sample_batch['phase'], sample_batch['chi']
        phase_batch = 1e6 * phase_batch.to(device).float()
        chi_batch = 1e6 * chi_batch.to(device).float()
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            chi_calc = model(phase_batch)
            loss = mse_loss(chi_calc, chi_batch) + lambda_reg * total_variation(chi_calc)
            loss.backward()
            optimizer.step()

        # statistics
        actual_loss += loss.item()
        echo_line('Train', epoch, batch_idx, phase_batch, n_dataset, actual_loss)
    train_loss[epoch] = actual_loss / tmp
    return model, train_loss


def val_phase(model, loader, mse_loss, lambda_reg, best_loss, epoch, val_loss, device):
    """
    Performs the validation phase of the training step
    :param device:
    :param model: Model to validate the training phase
    :param loader: Object to load the validation images
    :param mse_loss: Loss function
    :param lambda_reg:
    :param best_loss:
    :param epoch:
    :param val_loss:
    :return:
    """
    n_dataset = len(loader['val'].data)
    model.eval()
    actual_loss = 0.0
    tmp = 0

    # Iterate over data
    for batch_idx, sample_batch in enumerate(loader['val']):
        tmp += 1
        phase_batch, chi_batch = sample_batch['phase'], sample_batch['chi']
        phase_batch = 1e6 * phase_batch.to(device).float()
        chi_batch = 1e6 * chi_batch.to(device).float()

        with torch.set_grad_enabled(False):
            chi_calc = model(phase_batch)
            loss = mse_loss(chi_calc, chi_batch) + lambda_reg * total_variation(chi_calc)

        # statistics
        actual_loss += loss.item()
        echo_line('Val', epoch, batch_idx, phase_batch, n_dataset, actual_loss)
    val_loss[epoch] = actual_loss / tmp
    if actual_loss < best_loss:
        best_loss = actual_loss

    return model, val_loss, best_loss


@tf.function
def train_gan_phase(models, train_generator, optimizers, fn_losses, train_loss, actual_epoch, dataset_size):
    """
    Train phase of the cycle gan model. It gets the list of models composed by the generators and discriminators, also
    train generator of the dataset, the list of optimizers of each model, and returns the models and losses that
    generates each model.
    :param dataset_size: Size of the dataset
    :param models: Dictionary of tensorflow Models with the susceptibility and phase generators and discriminators
    :param train_generator: Batched generator with the train dataset
    :param optimizers: Dictionary of optimizers of each model
    :param fn_losses: Losses that defines the cycle gan paper of cycle gan
    :param train_loss: Dictionary of train losses with the number of epochs
    :param actual_epoch: Actual epoch of the training phase
    :return: models: Dictionary with the trained models
    :return: train_loss: List with the different losses for each Model in each epoch
    """
    generator_chi = models['gen_chi']
    generator_phi = models['gen_phi']
    discriminator_chi = models['disc_chi']
    discriminator_phi = models['disc_phi']

    opt_gen_chi = optimizers['gen_chi']
    opt_gen_phi = optimizers['gen_phi']
    opt_disc_chi = optimizers['disc_chi']
    opt_disc_phi = optimizers['disc_phi']

    actual_loss = {'gen_chi': 0.0, 'gen_phi': 0.0,
                   'disc_chi': 0.0, 'disc_phi': 0.0}

    for idx, sample_data in enumerate(train_generator):
        phi, chi = sample_data['phase'], sample_data['chi']
        phi = tf.cast(1e6 * phi, dtype=tf.double)
        chi = tf.cast(1e6 * chi, dtype=tf.double)
        with tf.GradientTape(persistent=True) as tapes:
            # Real outputs
            # out_phi = discriminator_phi.predict(phi)
            # out_chi = discriminator_chi.predict(chi)
            out_phi = discriminator_phi(phi)
            out_chi = discriminator_chi(chi)

            # Get the fake images, outputs and cycled images
            generated_chi = generator_chi(phi)
            out_generated_chi = discriminator_chi(generated_chi)
            cycled_phi = generator_phi(generated_chi)

            generated_phi = generator_phi(chi)
            out_generated_phi = discriminator_phi(generated_phi)
            cycled_chi = generator_chi(generated_phi)

            # Get identity images for identity loss
            same_chi = generator_chi(chi)
            same_phi = generator_phi(phi)

            #########################################
            # Calculate loss
            #########################################
            adversarial_loss_chi = fn_losses.generator_loss(out_generated_chi)
            adversarial_loss_phi = fn_losses.generator_loss(out_generated_phi)
            identity_loss_susceptibility = fn_losses.physics_loss(chi, same_chi, )
            identity_loss_phase = fn_losses.physics_loss(phi, same_phi, )
            recovered_loss_chi = fn_losses.supervised_loss(chi, cycled_chi, weight=0.6)
            recovered_loss_phi = fn_losses.supervised_loss(phi, cycled_phi, weight=0.6)

            loss_gen_chi = adversarial_loss_chi + identity_loss_susceptibility + recovered_loss_chi
            loss_gen_phi = adversarial_loss_phi + identity_loss_phase + recovered_loss_phi

            loss_disc_chi = fn_losses.discriminator_loss(out_chi, out_generated_chi)
            loss_disc_phi = fn_losses.discriminator_loss(out_phi, out_generated_phi)
        #########################################
        # Calculate gradients
        #########################################
        gradients_generator_chi = tapes.gradient(loss_gen_chi, generator_chi.trainable_variables)
        gradients_generator_phi = tapes.gradient(loss_gen_phi, generator_phi.trainable_variables)
        gradients_discriminator_chi = tapes.gradient(loss_disc_chi, discriminator_chi.trainable_variables)
        gradients_discriminator_phi = tapes.gradient(loss_disc_phi, discriminator_phi.trainable_variables)

        #########################################
        # Apply gradients
        #########################################
        opt_gen_chi.apply_gradients(zip(gradients_generator_chi, generator_chi.trainable_variables))
        opt_gen_phi.apply_gradients(zip(gradients_generator_phi, generator_phi.trainable_variables))
        opt_disc_chi.apply_gradients(zip(gradients_discriminator_chi, discriminator_chi.trainable_variables))
        opt_disc_phi.apply_gradients(zip(gradients_discriminator_phi, discriminator_phi.trainable_variables))
        actual_loss['gen_chi'] += loss_gen_chi
        actual_loss['gen_phi'] += loss_gen_phi
        actual_loss['disc_chi'] += loss_disc_chi
        actual_loss['disc_phi'] += loss_disc_phi

        echo_line('Chi generator train phase', actual_epoch, idx, phi, dataset_size, actual_loss['gen_chi'])
        echo_line('Phi generator train phase', actual_epoch, idx, phi, dataset_size, actual_loss['gen_phi'])
        echo_line('Chi discriminator train phase', actual_epoch, idx, phi, dataset_size, actual_loss['disc_chi'])
        echo_line('Phi discriminator train phase', actual_epoch, idx, phi, dataset_size, actual_loss['disc_phi'])

    models = {'gen_chi': generator_chi, 'gen_phi': generator_phi,
              'disc_chi': discriminator_chi, 'disc_phi': discriminator_phi}
    train_loss['gen_chi'][actual_epoch] = actual_loss['gen_chi']
    train_loss['gen_phi'][actual_epoch] = actual_loss['gen_phi']
    train_loss['disc_chi'][actual_epoch] = actual_loss['disc_chi']
    train_loss['disc_phi'][actual_epoch] = actual_loss['disc_phi']
    return models, train_loss


@tf.function
def val_gan_phase(models, val_generator, fn_losses, val_loss, actual_epoch, dataset_size):
    """
        Val phase of the cycle gan model. It gets the list of models composed by the generators and discriminators, also
        the val_generator of the dataset, and returns losses that generates each model.
        :param dataset_size: Size of the dataset
        :param models: Dictionary of tensorflow Models with the susceptibility and phase generators and discriminators
        :param val_generator: Batched generator with the val dataset
        :param fn_losses: Losses that defines the cycle gan paper of cycle gan
        :param val_loss: Dictionary of train losses with the number of epochs
        :param actual_epoch: Actual epoch of the training phase
        :return: train_loss: List with the different losses for each Model in each epoch
        """
    generator_chi = models['gen_chi']
    generator_phi = models['gen_phi']
    discriminator_chi = models['disc_chi']
    discriminator_phi = models['disc_phi']

    actual_loss = {'gen_chi': 0.0, 'gen_phi': 0.0,
                   'disc_chi': 0.0, 'disc_phi': 0.0}

    for idx, sample_data in enumerate(val_generator):
        phi, chi = sample_data['phase'], sample_data['chi']
        phi = tf.cast(1e6 * phi, dtype=tf.double)
        chi = tf.cast(1e6 * chi, dtype=tf.double)
        # Real outputs
        out_phi = discriminator_phi(phi)
        out_chi = discriminator_chi(chi)

        # Get the fake images, outputs and cycled images
        generated_chi = generator_chi(phi)
        out_generated_chi = discriminator_chi(generated_chi)
        cycled_phi = generator_phi(generated_chi)

        generated_phi = generator_phi(chi)
        out_generated_phi = discriminator_phi(generated_phi)
        cycled_chi = generator_chi(generated_phi)

        # Get identity images for identity loss
        same_chi = generator_chi(chi)
        same_phi = generator_phi(phi)

        #########################################
        # Calculate loss
        #########################################
        adversarial_loss_chi = fn_losses.generator_loss(out_generated_chi)
        adversarial_loss_phi = fn_losses.generator_loss(out_generated_phi)
        identity_loss_susceptibility = fn_losses.physics_loss(chi, same_chi, )
        identity_loss_phase = fn_losses.physics_loss(phi, same_phi, )
        recovered_loss_chi = fn_losses.supervised_loss(chi, cycled_chi, weight=0.6)
        recovered_loss_phi = fn_losses.supervised_loss(phi, cycled_phi, weight=0.6)

        loss_gen_chi = adversarial_loss_chi + identity_loss_susceptibility + recovered_loss_chi
        loss_gen_phi = adversarial_loss_phi + identity_loss_phase + recovered_loss_phi

        loss_disc_chi = fn_losses.discriminator_loss(out_chi, out_generated_chi)
        loss_disc_phi = fn_losses.discriminator_loss(out_phi, out_generated_phi)
        actual_loss['gen_chi'] += loss_gen_chi
        actual_loss['gen_phi'] += loss_gen_phi
        actual_loss['disc_chi'] += loss_disc_chi
        actual_loss['disc_phi'] += loss_disc_phi

        echo_line('Chi generator val phase', actual_epoch, idx, phi, dataset_size, actual_loss['gen_chi'])
        echo_line('Phi generator val phase', actual_epoch, idx, phi, dataset_size, actual_loss['gen_phi'])
        echo_line('Chi discriminator val phase', actual_epoch, idx, phi, dataset_size, actual_loss['disc_chi'])
        echo_line('Phi discriminator val phase', actual_epoch, idx, phi, dataset_size, actual_loss['disc_phi'])

    val_loss['gen_chi'][actual_epoch] = actual_loss['gen_chi']
    val_loss['gen_phi'][actual_epoch] = actual_loss['gen_phi']
    val_loss['disc_chi'][actual_epoch] = actual_loss['disc_chi']
    val_loss['disc_phi'][actual_epoch] = actual_loss['disc_phi']
    return models, val_loss


@tf.function
def training_step(model, optimizer, loss_fn, chi, phi, matrix_projection, loss_f):
    """
    Training step using Tensorflow
    :param matrix_projection: Matrix of the linear model of the susceptibility tensor
    :param phi: Input of the model corresponding to the bulk phase of the susceptibility tensor
    :param chi: Susceptibility tensor
    :param loss_f: Loss function for the physical model
    :param model: Model to be trained
    :param optimizer: Optimizer used
    :param loss_fn: Function to calculate the loss
    :return: model, train_loss
    """
    with tf.GradientTape() as tape:
        chi_model = model(phi, training=True)
        print(phi.shape)
        loss = loss_fn(chi, chi_model) + args.lambda_reg*loss_f(chi_model, phi, matrix_projection)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model, loss


@tf.function
def val_step(model, loss_fn, chi, phi, matrix_projection, loss_f):
    """
    Validation step using TensorFlow
    :param chi:
    :param phi:
    :param matrix_projection:
    :param loss_f: Loss function for the physical model
    :param model: Model to be trained
    :param loss_fn: Loss function
    :return: model, val_loss
    """
    chi_model = model(phi, training=False)
    loss = loss_fn(chi, chi_model) + args.lambda_reg*loss_f(chi_model, phi, matrix_projection)
    return model, loss

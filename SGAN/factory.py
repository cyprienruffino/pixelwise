from enum import Enum


class discriminators(Enum):
    classical_sgan = "classical_sgan"
    pix2pix_disc70x70 = "pix2pix_disc70x70"
    pix2pix_disc1x1 = "pix2pix_disc1x1"
    pix2pix_disc16x16 = "pix2pix_disc16x16"
    pix2pix_disc256x256 = "pix2pix_disc256x256"


class generators(Enum):
    classical_sgan = "classical_sgan"
    pix2pix_decoder = "pix2pix_decoder"
    pix2pix_encoder_decoder = "pix2pix_encoder_decoder"
    pix2pix_unet = "pix2pix_unet"
    pix2pix_patchgan = "pix2pix_patchgan"


class losses(Enum):
    classical_sgan = "classical_sgan"
    epsilon_gan = "epsilon_gan"
    softplus_gan = "softplus_gan"
    wasserstein_gan = "wasserstein_gan"
    wasserstein_min_gan = "wasserstein_min_gan"


class optimizer(Enum):
    adam = "adam"
    rmsprop = "rmsprop"
    sgd = "sgd"


def get_generator(gen):
    if gen == generators.classical_sgan:
        from generator import classical_sgan as _gen
    elif gen == generators.pix2pix_decoder:
        from generator import pix2pix_decoder as _gen
    elif gen == generators.pix2pix_encoder_decoder:
        from generator import pix2pix_encoder_decoder as _gen
    elif gen == generators.pix2pix_unet:
        from generator import pix2pix_unet as _gen
    elif gen == generators.pix2pix_patchgan:
        from generator import pix2pix_patchgan as _gen
    else:
        raise "Unknown generator " + gen

    return _gen


def get_discriminator(disc):
    if disc == discriminators.classical_sgan:
        from discriminator import classical_sgan as _disc
    elif disc == discriminators.pix2pix_disc70x70:
        from discriminator import pix2pix_disc70x70 as _disc
    elif disc == discriminators.pix2pix_disc1x1:
        from discriminator import pix2pix_disc1x1 as _disc
    elif disc == discriminators.pix2pix_disc16x16:
        from discriminator import pix2pix_disc16x16 as _disc
    elif disc == discriminators.pix2pix_disc256x256:
        from discriminator import pix2pix_disc256x256 as _disc
    else:
        raise "Unknown discriminator " + disc

    return _disc


def get_losses(loss):
    if loss == losses.classical_sgan:
        from losses import gan_true as loss_true
        from losses import gan_fake as loss_fake
        from losses import gan_gen as loss_gen
    elif loss == losses.epsilon_gan:
        from losses import epsilon_gan_true as loss_true
        from losses import epsilon_gan_fake as loss_fake
        from losses import epsilon_gan_gen as loss_gen
    elif loss == losses.wasserstein_gan:
        from losses import wasserstein_true as loss_true
        from losses import wasserstein_fake as loss_fake
        from losses import wasserstein_gen as loss_gen
    elif loss == losses.wasserstein_min_gan:
        from losses import wasserstein_min_true as loss_true
        from losses import wasserstein_min_fake as loss_fake
        from losses import wasserstein_min_gen as loss_gen
    elif loss == losses.softplus_gan:
        from losses import softplus_gan_true as loss_true
        from losses import softplus_gan_fake as loss_fake
        from losses import softplus_gan_gen as loss_gen
    else:
        raise "Unknown losses " + loss

    return loss_true, loss_fake, loss_gen


def get_optimizer(optimizer, config):
    if optimizer == optimizer.adam:
        from keras.optimizers import Adam
        _optimizer = Adam(lr=config.lr, beta_1=config.b1)
    elif optimizer == optimizer.rmsprop:
        from keras.optimizers import RMSProp
        _optimizer = RMSProp(lr=config.lr)
    elif optimizer == optimizer.sgd:
        from keras.optimizers import SGD
        _optimizer = SGD(lr=config.lr, momentum=config.momentum)
    else:
        raise "Unknown optimizer " + optimizer
    return _optimizer

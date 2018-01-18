#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 3D version of the 2D SGAN (Jetchev et al., 2016) by Eric Laloy <elaloy@sckcen.be>


import keras.backend as K
import numpy as np
from config3D import Config
from data_io3D import save_tensor
from keras.engine import Model
from keras.initializers import RandomNormal, Constant
from keras.layers import (BatchNormalization,
                          GaussianNoise, Input, LeakyReLU, Conv3DTranspose, Conv3D)
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from tools import TimePrint


w_init = RandomNormal(stddev=0.02)
b_init = Constant(value=0.0)
g_init = RandomNormal(mean=1., stddev=0.02)


def obj_d(y_true, y_pred):
    return - K.mean(K.log(1 - y_pred[1])) \
           - K.mean(K.log(y_pred[0]))  # The regularization term is added in the layers


def obj_g(y_true, y_pred):
    return -K.mean(K.log(y_pred))


##
# network code
class SGAN(object):

    def __init__(self, name=None):
        '''
        @static configuration class
        @param name     load stored sgan model
        '''
        self.config = Config
        if name is not None:
            print("loading parameters from file:", name)

            self.model_d = load_model(name+"_d.hdf5", custom_objects={"obj_d": obj_d, "obj_g": obj_g})
            self.model_g = load_model(name + "_g.hdf5", custom_objects={"obj_d": obj_d, "obj_g": obj_g})
            self.model_gen = load_model(name + "_gen.hdf5", custom_objects={"obj_d": obj_d, "obj_g": obj_g})

            TimePrint("Compiling the network...\n")
            self.model_d.compile(optimizer=Adam(lr=self.config.lr, beta_1=self.config.b1),
                                 loss=obj_d)
            TimePrint("Discriminator done.")
            self.model_g.compile(optimizer=Adam(lr=self.config.lr, beta_1=self.config.b1),
                                 loss=obj_g)
            TimePrint("Generator done.")

            self.generate = self.model_g.predict
            self.train_g = lambda Znp: self.model_g.train_on_batch(Znp, np.zeros((Znp.shape)))
            self.train_d = lambda samples, Znp: self.model_d.train_on_batch([samples, Znp], [np.zeros(samples.shape),
                                                                                             np.zeros(Znp.shape)])



        else:
            self._setup_gen_params(self.config.gen_ks, self.config.gen_fn)
            self._setup_dis_params(self.config.dis_ks, self.config.dis_fn)
            self._build_sgan()

    def save(self, name):
        print("saving SGAN parameters in file: ", name)

        self.model_d.save(name + "_d.hdf5")
        self.model_g.save(name + "_g.hdf5")
        self.model_gen.save(name + "_gen.hdf5")



    def _setup_gen_params(self, gen_ks, gen_fn):
        '''
        set up the parameters, i.e. filter sizes per layer and depth, of the generator
        '''
        ## 
        # setup generator parameters and sanity checks
        if gen_ks==None:
            self.gen_ks = [(5,5,5)] * 5   # set to standard 5-layer net
        else:
            self.gen_ks = gen_ks

       
        self.gen_depth = len(self.gen_ks)

        if gen_fn!=None:
            assert len(gen_fn)==len(self.gen_ks), 'Layer number of filter numbers and sizes does not match.'
            self.gen_fn = gen_fn
        else:
            self.gen_fn = [64] * self.gen_depth
    

    def _setup_dis_params(self, dis_ks, dis_fn):
        '''
        set up the parameters, i.e. filter sizes per layer and depth, of the discriminator
        '''
        ##
        # setup discriminator parameters
        if dis_ks==None:
            self.dis_ks = [(5,5,5)] * 5   # set to standard 5-layer net
        else:
            self.dis_ks = dis_ks

        self.dis_depth = len(dis_ks)

        if dis_fn!=None:
            assert len(dis_fn)==len(self.dis_ks), 'Layer number of filter numbers and sizes does not match.'
            self.dis_fn = dis_fn
        else:
            self.dis_fn = [64] * self.dis_depth


    def _spatial_generator(self, layer):
        for l in range(self.gen_depth - 1):
            tconv = Conv3DTranspose(
                filters=self.gen_fn[l],
                kernel_size=self.gen_ks[l],
                activation="relu",
                kernel_regularizer=l2(self.config.l2_fac),
                data_format="channels_first")(layer)
            layer = BatchNormalization(
                gamma_initializer=g_init,
                beta_initializer=b_init,
                axis=1)(tconv)

        output = Conv3DTranspose(
            filters=self.gen_fn[-1],
            kernel_size=self.gen_ks[-1],
            activation="tanh",
            kernel_regularizer=l2(self.config.l2_fac),
            data_format="channels_first")(layer)

        return output

    def _spatial_discriminator(self, layer):
        noise = GaussianNoise(stddev=0.1)(layer)
        layer = Conv3D(
            filters=self.dis_fn[0],
            kernel_size=self.dis_ks[0],
            activation="linear",
            kernel_regularizer=l2(self.config.l2_fac),
            data_format="channels_first")(noise)
        layer = LeakyReLU()(layer)

        for l in range(1, self.dis_depth - 1):
            conv = Conv3D(
                filters=self.dis_fn[l],
                kernel_size=self.dis_ks[l],
                activation="linear",
                kernel_regularizer=l2(self.config.l2_fac),
                data_format="channels_first")(layer)
            layer = LeakyReLU()(conv)
            layer = BatchNormalization(
                gamma_initializer=g_init,
                beta_initializer=b_init,
                axis=1)(layer)

        output = Conv3D(
            filters=self.dis_fn[-1],
            kernel_size=self.dis_ks[-1],
            activation="sigmoid",
            kernel_regularizer=l2(self.config.l2_fac),
            data_format="channels_first")(layer)

        return output




    def _build_sgan(self):
        Z = Input((self.config.nz, None, None, None))
        X = Input((self.config.nc, self.config.npx,
                   self.config.npx, self.config.npx))

        gen_X = self._spatial_generator(Z)
        d_real = self._spatial_discriminator(X)
        d_fake = self._spatial_discriminator(gen_X)

        self.model_gen = Model(inputs=Z, outputs=gen_X)
        self.model_d = Model(inputs=[X, Z], outputs=[d_real, d_fake])
        self.model_g = Model(inputs=Z, outputs=d_fake)

        TimePrint("Compiling the network...\n")
        self.model_d.compile(optimizer=Adam(lr=self.config.lr, beta_1=self.config.b1),
                             loss=obj_d)
        TimePrint("Discriminator done.")
        self.model_g.compile(optimizer=Adam(lr=self.config.lr, beta_1=self.config.b1),
                             loss=obj_g)
        TimePrint("Generator done.")

        self.generate = self.model_g.predict
        self.train_g = lambda Znp: self.model_g.train_on_batch(Znp, np.zeros((Znp.shape)))
        self.train_d = lambda samples, Znp: self.model_d.train_on_batch([samples, Znp], [np.zeros(samples.shape), np.zeros(Znp.shape)])

        self.model_d.summary()
        # plot_model(self.model_d, to_file="model_d.png") # Draw the network graph in a .png file
        # plot_model(self.model_g, to_file="model_g.png")

if __name__=="__main__":

    c           = Config

    if c.load_name   == None:
        sgan        = SGAN()
    else:
        sgan        = SGAN(name='models/' + c.load_name)
        
    c.print_info()

    ##
    # 
    z_sample = np.random.uniform(-1., 1., (1, c.nz, c.zx_sample, c.zx_sample, c.zx_sample))
    epoch           = 0
    tot_iter        = 0
    
    while (epoch<=c.num_epoch):
        epoch       += 1
        print("Epoch %d" % epoch)

        Gcost = []
        Dcost = []

        iters = c.epoch_iters / c.batch_size
        for it, samples in enumerate(c.data_iter):
            if it >= iters:
                break
            tot_iter+=1

            Znp = np.random.uniform(-1., 1., (c.batch_size, c.nz, c.zx, c.zx, c.zx))

            if tot_iter % (c.k+1) == 0:
                cost = sgan.train_g(Znp)
                Gcost.append(cost)
            else:
                cost = sgan.train_d(samples,Znp)
                Dcost.append(cost)

        print("Gcost=", np.mean(Gcost), "  Dcost=", np.mean(Dcost))

        data = sgan.generate(z_sample)

        save_tensor(data[0], 'samples/%s_epoch%d' % (c.save_name,epoch))
        sgan.save('models/%s_epoch%d.sgan'%(c.save_name,epoch))


import os
from tools import create_dir
from data_io3D import get_texture_iter


create_dir('samples')               # create, if necessary, for the output samples 
create_dir('models') 


def zx_to_npx(zx, depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    # note: in theano we'd have zx*2**depth
    return (zx - 1)*2**depth + 1


class Config(object):
    '''
    wraps all configuration parameters in 'static' variables
    '''
    ##
    # network parameters
    nz          = 3                  # num of dim for Z at each field position (referred to as d in the paper?) 
    zx          = 3 #4                   # number of spatial dimensions in Z (referred to as l and m in the paper?) 
    zx_sample   = 5 #7                    # size of the spatial dimension in Z for producing the samples
	# l = h/r, m = w/r
    nc          = 1                     # number of channels in input X (i.e. r,g,b) #3
    n_conv_layer = 2 #4
    # kernel sizes on each layer - should be odd numbers for zero-padding stuff:       
    # gen_ks      = ([(3,3,3)] * n_conv_layer)[::-1]  # MCA TI
    # dis_ks      = [(3,3,3)] * n_conv_layer   # MCA TI
    gen_ks      = ([(5,5,5)] * n_conv_layer)[::-1]  # Continuous Fold TI
    dis_ks      = [(5,5,5)] * n_conv_layer   # Continuous Fold TI
    gen_ls      = len(gen_ks)           # num of layers in the generative network
    dis_ls      = len(dis_ks)           # num of layers in the discriminative network
    #gen_fn      = [nc]+[2**(n+6) for n in range(gen_ls-1)]  # generative number of filters # MCA TI
    gen_fn      = [nc]+[2**(n+6) for n in range(gen_ls-1)]
    gen_fn      = gen_fn[::-1]
    #dis_fn      = [2**(n+6) for n in range(dis_ls-1)]+[1]   # discriminative number of filters # MCA TI
    dis_fn      = [2**(n+6) for n in range(dis_ls-1)]+[1]

    lr          = 0.0005                # learning rate of adam
    b1          = 0.5                   # momentum term of adam
    l2_fac      = 1e-5                  # L2 weight regularization factor

    batch_size  = 25 
	#batch_size  = 64

    epoch_iters = batch_size * 100
    num_epoch=200

    k           = 1                     # number of D updates vs G updates

    npx         = zx_to_npx(zx, gen_ls) # num of pixels width/height/depth of 3D images in X

    ##
    # data input folder
    sub_name    = 'ti_3D' # 3D training image
    home        = os.path.expanduser("~")
    texture_dir = home + "/labo-litis/cypricyp/spatial_gan/keras/3D/%s/" % sub_name
    data_iter   = get_texture_iter(texture_dir, npx=npx, mirror=False, batch_size=batch_size,n_channel=nc)

    save_name   = sub_name+ "_filters%d_npx%d_%dgL_%ddL" % (dis_fn[0],npx,gen_ls, dis_ls)

    load_name   = None                  # if None, initializing network from scratch
    #load_name   = "ti3Dcont_filters64_npx65_3gL_3dL_epoch10.sgan"


    @classmethod
    def print_info(cls):
        ##
        # output some information
        print("Learning and generating samples from zx ", cls.zx, ", which yields images of size npx ", zx_to_npx(cls.zx, cls.gen_ls)) 
        print("Producing samples from zx_sample ", cls.zx_sample, ", which yields images of size npx ", zx_to_npx(cls.zx_sample, cls.gen_ls)) 
        print("Saving samples and model data to file ", cls.save_name)


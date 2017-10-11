#!/usr/bin/env python2

import sys
import os
import imp
import scipy, scipy.misc
import numpy as np
from  scipy.signal import medfilt
import h5py
import matplotlib.pyplot as plt
import time

os.chdir('/home/elaloy/spatial_gan')

sys.path.append('../spatial_gan')
from sgan import SGAN

modelpath = '/home/elaloy/spatial_gan/models4_/ti_used_filters64_npx353_5gL_5dL_epoch23.sgan'
configpath='/home/elaloy/spatial_gan/models4_/config.py'

TriCatTI=False
DoFiltering=True
DoThreshold=True

def main_test_local():
    config = imp.load_source('config', configpath)
    sgan        = SGAN(modelpath)
    c=config.Config
    # c = sgan.config
    #c.print_info()
    
    np.random.seed(0)
    z_sample1= np.random.uniform(-1.,1., (1, c.nz, c.zx_sample-10, c.zx_sample-10) )
    np.save('z0.npy',z_sample1)
    
    data = sgan.generate(z_sample1)
    np.save('x0.npy',data)
    scipy.misc.imsave('out0.png', np.array(data[0,0,:,:]))
    
    z_sample2=np.array(z_sample1)
    z_sample2[0,0,-1,-1,-1]=-1* z_sample2[0,0,-1,-1]
    data = sgan.generate(z_sample2)
    np.save('x2.npy',data)
    scipy.misc.imsave('out2.png', np.array(data[0,0,:,:]))

def main_gen(DoFiltering=DoFiltering,DoThreshold=DoThreshold,nsample=1,TriCatTI=TriCatTI):  
    config = imp.load_source('config', configpath)
    sgan        = SGAN(modelpath)
    c=config.Config
    #c = sgan.config
    t_start=time.time()
    np.random.seed(2046) # seed is 2046 for generating the realizations in the paper, seed is 123 for generating the true model in the inversions
    #z_sample1= np.random.uniform(-1.,1., (nsample, c.nz, c.zx, c.zx) )   
    z_sample1= np.random.uniform(-1.,1., (nsample, c.nz, 10, 10) )   
   
    model = sgan.generate(z_sample1)[:,0,:,:]
    
    model = (model+1)*0.5 # Convert from [-1,1] to [0,1]
   
    if DoFiltering==True:
        
        for ii in xrange(model.shape[0]):
            model[ii,:]=medfilt(model[ii,:], kernel_size=(3,3))
            
    if DoThreshold and not(TriCatTI):
#        for ii in xrange(model.shape[0]):
#            threshold=filters.threshold_otsu(model[ii,:])
#            model[ii,:][model[ii,:]<threshold]=0
#            model[ii,:][model[ii,:]>=threshold]=1
        threshold=0.5
        model[model<threshold]=0
        model[model>=threshold]=1
        
    if DoThreshold and TriCatTI:
        model[model<0.334]=0
        model[model>=0.667]=2
        model[np.where((model > 0) & (model < 2))]=1
        model=model/2.0 
     
    
    print('elapsed_time is: ',time.time()-t_start) 
    plt.figure(figsize=(8,8)) 
    plt.imshow(model[0,:,:],cmap='gray')
    
    h5_filename="2D_Gen_" +'_'+str(c.zx)
    f = h5py.File(h5_filename+'.hdf5', mode='w')
    h5dset = f.create_dataset('features', data=model)
    f.flush()
    f.close()
    
if __name__=="__main__":
    main_gen()

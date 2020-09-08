#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 18:11:08 2020

@author: Daniele Mazzola 
"""
import numpy as np
import poppy
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import os
import time

#logging.basicConfig(level=logging.DEBUG) # more informative text output

#%% ---------------------------------- DATA PATHS ------------------------------------------------------

save_dir = os.path.join('..','..','TrainingSetZernike','test0','rr')

# If you want (change boolean):
normalisation = False
crop = False

#%% ----------------------------------- FUNCTIONS -----------------------------------------------------

# Random generation of Zernike coefficients
def generate_coefficients(wavelenght, wfe_budget):
    coefficients = []
    wavelength_m = wavelength.value * 1e-9
    for term in wfe_budget:
        # terms must be in m
        coefficients.append( wavelength_m / np.random.uniform(low = - term, high = + term) )
    return coefficients

# Cut the image in half (at the center)
def crop_in_center(image, crop_size):
  n_x = image.shape[0]
  n_y = image.shape[1]
  l = crop_size // 2
  x_center, y_center = n_x // 2 + 1, n_y // 2 + 1
  cropped_image = image[x_center - l : x_center + l, y_center - l : y_center + l]
  return cropped_image

def init_h5(save_dir):
    create_saving_directory(save_dir)
    timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    name = f'TrainingSet_{timestamp}.h5'
    filename = os.path.join(save_dir,name)
    f = h5py.File(filename, 'w', libver='latest')
    group = f.create_group('zernike_psf')  # Group 1
    return f, group

def create_saving_directory(save_path):        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)


#%% ----------------------------------- PARAMETERS --------------------------------------------------

# Input parameters
SAMPLING = 0.25               # beam ratio (See: documentation)
wavelength = 488 * u.nm
NA = 0.3                      # Numerical Aperture
fl_obj = 20 * u.mm            # objective focal length
fl_tube = 200 * u.mm          # tube focal length
delta = 0 * u.um              # defocus (at detector)
pix_size = 0.1 * u.um         # detector pixel sizes
n_pix = 256                   # pixels per side (detector)

crop_size = int(n_pix / 2)

# Other parameters
radius = NA * fl_obj          # Aperture stop radius

# Zernike coefficients (in meters)
n_coeff = 21
n_set = 5                                 # number of images generated in the set

#%% Zernike coefficients as lambda fraction
wf_error_budget = [100]*n_coeff
wf_error_budget[0] = 1e15
wf_error_budget[1] = 1e15
wf_error_budget[2] = 1e15
# wf_error_budget[3] = 10
# wf_error_budget[4] = 10
# wf_error_budget[5] = 10
# wf_error_budget[6] = 10
# wf_error_budget[7] = 10
# wf_error_budget[8] = 10
# wf_error_budget[9] = 10
# wf_error_budget[10] = 10
# wf_error_budget[11] = 10
# wf_error_budget[12] = 10
# wf_error_budget[13] = 10
# wf_error_budget[14] = 10
# wf_error_budget[15] = 10
# wf_error_budget[16] = 10
# wf_error_budget[17] = 10
# wf_error_budget[18] = 10
# wf_error_budget[19] = 10
# wf_error_budget[20] = 10


#%% ----------------------------------- OPTICAL ELEMENTS ----------------------------------------------

aperture = poppy.CircularAperture(radius=radius, name='Pupil')

objective = poppy.QuadraticLens(fl_obj, name = 'Objective lens')

# ----------------------------------- OPTICAL SYSTEM ------------------------------------------------

# Initialize results as arrays
psf_results = []
zernike_coeff = []

file_h5, group_h5 = init_h5(save_dir)

for image_idx in range(n_set):

    osys = poppy.FresnelOpticalSystem(pupil_diameter=10*radius, npix=256, beam_ratio=SAMPLING)
    
    osys.add_optic(aperture) # The system pupil
    
    coefficients = generate_coefficients(wavelength, wf_error_budget)
    zernike_wfe = poppy.ZernikeWFE(radius = radius, coefficients = coefficients,
                             name = 'Aberrated Wavefront')
    
    osys.add_optic(zernike_wfe)
    
    osys.add_optic(objective, distance=fl_obj)
    
    osys.add_detector(pixelscale = pix_size / u.pixel, fov_pixels=n_pix, distance=fl_obj+delta)
    
    # System description just one time
    if image_idx == 1:
          print('\nOptical System description:')
          osys.describe()
          print('\n')
          
    
    #plt.figure(figsize=(12,8))
    psf = osys.calc_psf(wavelength=wavelength, display_intermediates=False, return_intermediates=False)
    
    plt.figure(figsize=(10,10))
    poppy.display_psf(psf, normalize = 'total')
    plt.title('objective lens PSF')
    
    if crop:
          
          psf_train = crop_in_center(psf[0].data, crop_size=crop_size)
    else:
          psf_train = psf[0].data
          
    
    if normalisation:
          
          psf_train = psf_train / np.max(psf_train)
          
    
    # Update results 
    
    plt.figure(figsize=(10,10))
    plt.imshow(psf_train)
    plt.title('objective lens PSF - cropped')
    plt.show()      
                   
    #%% Save results in h5 format
    
    dataset_name = f't0/c0/aberrated_PSF_{image_idx:05d}'
    attribute_name = f'coefficients_{image_idx:05d}'
                
    data_h5 = group_h5.create_dataset(name = dataset_name,
                                      data = psf_train)
    data_h5.attrs[attribute_name] = coefficients
    data_h5.flush()
    
file_h5.close()
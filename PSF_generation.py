#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 18:11:08 2020

@author: Daniele Mazzola 
"""
import numpy as np
from scipy import signal
import poppy
import astropy.units as u
import matplotlib.pyplot as plt
import h5py
import os
import time

#logging.basicConfig(level=logging.DEBUG) # more informative text output

#%% ---------------------------------- DATA PATHS ------------------------------------------------------

save_dir = os.path.join('..','..','Training_Set_Zernike','test_0','rr')

# If you want (change boolean):
normalisation = False
crop = False

#%% ----------------------------------- FUNCTIONS -----------------------------------------------------

# Random generation of Zernike coefficients
def generate_coefficients(wavelenght, wfe_budget):
    coefficients = []
    for term in wfe_budget:
        # Fractions of lambda (terms must be in m)
        coefficients.append( np.random.uniform(low = - term, high = + term) )
    return coefficients

# Distributions of lambda fractions for Zernike coefficients:
def sigmoid_budget(n_coeff):
      # Lambda in m
      wavelength_m = wavelength.value * 1e-9
      # Sigmoid parameters
      n_points = 4 * n_coeff
      x = np.linspace(- n_coeff, + n_coeff, n_points)
      ampl_0 = 1e3
      shift_sigm = 2
      # Sigmoid points for wf_error_budget (lambda fractions)
      sigmoid = ampl_0 * (1 / (1 + np.exp( - 0.5 * x ))) + shift_sigm
      sigmoid_sampled = wavelength_m / [round(value) for value in sigmoid[0 : n_points : 4]]
      sigmoid_sampled[:3] = 0 # no piston and x,y tilts
      return sigmoid_sampled

def gaussian_budget(n_coeff):
      # Lambda in m
      wavelength_m = wavelength.value * 1e-9
      # Gaussian parameters
      n_points = 4 * n_coeff
      standard_deviation = round(n_coeff / 2)
      ampl_0 = 1e3
      shift_gauss = 1
      # Gaussian
      gaussian = ampl_0 * signal.gaussian(M = n_points, std = standard_deviation) + shift_gauss
      # Gaussian raising edge points for wf_error_budget (lambda fractions)
      gauss_sampled = wavelength_m / [round(value) for value in gaussian[0 : int(n_points/2) : 2]]
      gauss_sampled[: 3] = 0 # no piston and x,y tilts
      return gauss_sampled

def exponential_budget(n_coeff):
      # Lambda in m
      wavelength_m = wavelength.value * 1e-9
      # Exp parameters
      n_points = 4 * n_coeff
      x = np.linspace(0, n_coeff, n_points)
      ampl_0 = 0.2
      # Shifted exponential
      shift_exp = 0
      exponential = ampl_0 * np.exp(x) + shift_exp
      # Exponential points for wf_error_budget (lambda fractions)
      exp_sampled = wavelength_m / [round(value) for value in exponential[0 : n_points : 4]]
      exp_sampled[:3] = 0 # no piston and x,y tilts
      return exp_sampled

# Cut the image in half (in center)
def crop_in_center(image, crop_size):
  n_x = image.shape[1]
  n_y = image.shape[0]
  l = crop_size // 2
  x_center, y_center = n_x // 2 + 1, n_y // 2 + 1
  cropped_image = image[x_center - l : x_center + l, y_center - l : y_center + l]
  return cropped_image

# Create path if it does not exist yet
def create_saving_directory(save_path):        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            
# Create h5 file            
def init_h5(save_dir):
    create_saving_directory(save_dir)
    timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    name = f'TrainingSet_{timestamp}.h5'
    filename = os.path.join(save_dir,name)
    f = h5py.File(filename, 'w')
    group = f.create_group('zernike_psf')  # 1 group
    return f, group

#%% ----------------------------------- PARAMETERS --------------------------------------------------

# Input parameters
SAMPLING = 0.25               # beam ratio (See: documentation)
wavelength = 488 * u.nm
NA = 0.3                      # Numerical Aperture
fl_obj = 20 * u.mm            # objective focal length
delta = 0 * u.um              # defocus (at detector)
pix_size = 0.1 * u.um         # detector pixel sizes
n_pix = 256                   # pixels per side (square detector)


# Other parameters
radius = NA * fl_obj          # Aperture stop radius
crop_size = int(n_pix / 2)

# Zernike coefficients
n_coeff = 20
n_set = 10                    # number of images in the set

#%% Zernike coefficients distribution as lambda fractions

wf_error_budget = [] * n_coeff

print('Select desired coefficient distribution (default Sigmoid):')
print('Sigmoid: 1 \nGaussian: 2 \nExponential: 3 \n')
distribution = input()

if distribution == 2:
      wf_error_budget = gaussian_budget(n_coeff)
elif distribution == 3:
      wf_error_budget = exponential_budget(n_coeff)
else:
      wf_error_budget = sigmoid_budget(n_coeff)
      
#%% --------------------------------- OPTICAL ELEMENTS ----------------------------------------------

aperture = poppy.CircularAperture(radius=radius, name='Pupil')

objective = poppy.QuadraticLens(fl_obj, name = 'Objective lens')

# ----------------------------------- OPTICAL SYSTEM ------------------------------------------------

# Initialize results as lists
psf_results = []
zernike_coeff = []

file_h5, group_h5 = init_h5(save_dir)

for image_idx in range(n_set):

    osys = poppy.FresnelOpticalSystem(pupil_diameter=10*radius, npix=256, beam_ratio=SAMPLING)
    
    osys.add_optic(aperture) # The system pupil
    
    coefficients = generate_coefficients(wavelength, wf_error_budget)
    zernike_wfe = poppy.ZernikeWFE(radius = radius, coefficients = coefficients,
                             name = 'Aberrated Wavefront')
    
    osys.add_optic(zernike_wfe) # Non ideality
    
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
    plt.title('objective lens PSF_{},'.format(image_idx))
    
    if crop:
          
          psf_train = crop_in_center(psf[0].data, crop_size=crop_size)
    else:
          psf_train = psf[0].data
          
    
    if normalisation:
          
          psf_train = psf_train / np.max(psf_train)
          
    
    # Show results
    plt.figure(figsize=(10,10))
    plt.imshow(psf_train)
    plt.title('objective lens PSF_cropped_{}'.format(image_idx))
    plt.show()      
                   
    # Save results in h5 format
    dataset_name = f't0/c0/aberrated_PSF_{image_idx:05d}'
    attribute_name = f'coefficients_{image_idx:05d}'
                
    data_h5 = group_h5.create_dataset(name = dataset_name, data = psf_train)
    data_h5.attrs[attribute_name] = coefficients
    data_h5.flush()
    
file_h5.close()

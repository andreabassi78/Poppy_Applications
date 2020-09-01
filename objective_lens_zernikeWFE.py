# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:02:05 2020

@author: Andrea Bassi
"""

import poppy
import astropy.units as u
import matplotlib.pyplot as plt

radius = 6 * u.mm
f = 20 * u.mm

coefficients_sequence = [0.0, 0.0, 0.0, 0, 0.0, 0, 0.0, 0.0, 0.0] * u.um

osys = poppy.FresnelOpticalSystem(pupil_diameter = 2*radius, npix=256, beam_ratio=0.25)

osys.add_optic(poppy.CircularAperture(radius=radius, name='pupil')) # The system pupil

zernikewfe = poppy.ZernikeWFE(radius=radius, coefficients=coefficients_sequence)
osys.add_optic(zernikewfe)

m1 = poppy.QuadraticLens(f, name = 'objective lens')
osys.add_optic(m1, distance=f)

#osys.add_optic(poppy.CircularAperture(radius=diam/2, name='aperture0'), distance = 0*u.mm)
#m2 = poppy.QuadraticLens(fl_sec, name='Secondary')
#osys.add_optic(m2, distance=d_pri_sec)
#osys.add_optic(poppy.ScalarTransmission(name='free space'), distance=f1);

delta = 0* u.um

osys.add_detector(pixelscale=0.1*u.um/u.pixel, fov_pixels=256, distance = f+delta)

osys.describe()

plt.figure(figsize=(12,8))
psf, waves = osys.calc_psf(wavelength=0.5e-6, display_intermediates=True, return_intermediates=True)

plt.figure(figsize=(12,8))
poppy.display_psf(psf, normalize = 'total')
plt.title('objective lens PSF')

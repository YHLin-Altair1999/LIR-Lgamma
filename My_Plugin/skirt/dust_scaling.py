import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as c
from ..enclose import find_minimal_enclosing_radius_kdtree
from ..LoadData import get_center, get_angular_momentum, get_snap_path, get_radius
from glob import glob
import logging
from gizmo_analysis import gizmo_star
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def dust_to_gas_ratio_RemyRuyer(nO_nH_ratio):
    '''
    From Rémy-Ruyer et al. (2014): https://www.aanda.org/articles/aa/full_html/2014/03/aa22803-13/aa22803-13.html
    Using the broken power-law fit with X_{CO, Z}
    '''
    x = 12 + np.log10(nO_nH_ratio)

    a, b = 2.21, 0.96
    alpha_H, alpha_L = 1.00, 3.10
    x_t, x_sun = 8.10, 8.69
    
    y = (x > x_t)*(a + alpha_H*(x_sun - x)) + (x <= x_t)*(b + alpha_L*(x_sun - x))
    G_D_ratio = 10**y
    return 1/G_D_ratio

def dust_to_gas_ratio_Galliano(nO_nH_ratio):
    '''
    From Galliano et al. (2021): https://ui.adsabs.harvard.edu/abs/2021A%26A...649A..18G/abstract
    Using 4th order polynomial fit to the dust-to-gas ratio as a function of the oxygen abundance
    '''
    x = 12 + np.log10(nO_nH_ratio)
    log10_Z_dust = (
        (x >= 7.3) * (11471.808 - 5669.5959*x + 1045.9713*x**2 - 85.434332*x**3 + 2.6078774*x**4) + \
        (x <  7.3) * (-13.2 + x)
        )

    max_Z_x = np.argmax(log10_Z_dust)
    log10_Z_dust[max_Z_x:] = log10_Z_dust[max_Z_x]

    Z_dust = 10**log10_Z_dust
    return Z_dust


import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as c
from My_Plugin.skirt.convert import convert_stellar, convert_gas

def convert_gizmo_to_skirt()
if __name__ == '__main__':
    box_size = 50*u.kpc
    convert_stellar(snap_id=600, r_max=2**0.5*box_size)
    convert_gas(    snap_id=600, r_max=2**0.5*box_size)





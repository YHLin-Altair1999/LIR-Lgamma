import numpy as np
import matplotlib.pyplot as plt
import h5py
import astropy.units as u
import astropy.constants as c
import logging
import os
from My_Plugin.LoadData import get_center, get_angular_momentum, get_snap_path, get_radius
from glob import glob
from gizmo_analysis import gizmo_star

logging.basicConfig(level=logging.INFO)
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    'font.family': 'serif'
    })

def get_units(f):
    z = f['Header'].attrs.get('Redshift')
    h = f['Header'].attrs.get('HubbleParam')

    a = 1/(1+z) # the scale factor

    # Check http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#units
    code_mass = 1e10*u.M_sun/h
    code_length = 1e0*u.kpc*a/h
    code_velocity = 1*u.km/u.s*np.sqrt(a)
    #print(f"loading snapshot at z={z:.1f}, h={h:.2f}")
    #print(f"code mass: {code_mass}, code length: {code_length}, code velocity:{code_velocity}")
    return [code_mass, code_length, code_velocity]

def get_data(galaxy, snap=600):
    # Load the file
    target = os.path.join(get_snap_path(galaxy, snap), '*.hdf5')
    print('The target folder is', target)
    fnames = list(glob(target))
    print('The hdf5 files are', fnames)
    fs = [h5py.File(fname, 'r') for fname in fnames]
    return fs

def align_axis(galaxy: str, snap_id: int, array: np.ndarray) -> np.ndarray:
    L = np.array(get_angular_momentum(galaxy, snap_id))
    L /= np.linalg.norm(L)
    new_z = L
    # Find new x-axis: choose any vector perpendicular to new_z
    # We can use cross product with [1,0,0] or [0,1,0], whichever is not parallel to new_z
    if abs(np.dot(new_z, [1, 0, 0])) < 0.9:
        new_x = np.cross([1, 0, 0], new_z)
    else:
        new_x = np.cross([0, 1, 0], new_z)
    new_x = new_x / np.linalg.norm(new_x)

    # Find new y-axis using cross product
    new_y = np.cross(new_z, new_x)
    # new_y is already normalized since new_z and new_x are orthonormal

    # Create rotation matrix where columns are the new basis vectors
    rotation_matrix = np.column_stack([new_x, new_y, new_z])
    array[:,:3] = array[:,:3] @ rotation_matrix
    return array

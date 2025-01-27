import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yt
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simpson
from .LoadData import get_center
from .Add_Fields import add_fields
import astropy.units as u
import astropy.constants as c

def L_IR(fname, band=np.array([8,1000])*u.micron):
    '''
    Calculate the total IR luminosity from SKIRT-generated SED files.
    '''
    names = ['wavelength', 'total', 'transparent', 'direct', 'scattered', 'direct secondary', 'scattered secondary', 'transparent secondary']
    df = pd.read_csv(fname, names=names, skiprows=9, sep=' ')
    d = 1*u.Mpc
    F_nu = np.array(df['total'])*u.Jy
    L_nu = 4*np.pi*d**2*F_nu
    wav = np.array(df['wavelength'])*u.micron
    nu = c.c/wav
    L_IR = -simpson(L_nu.to('erg/(s*Hz)').value, x=nu.to('Hz').value)*u.erg/u.s
    return L_IR

def L_gamma(ds):
    ds = add_fields(ds)
    snap_id = int(ds.filename.split('/')[-1].split('.')[0].split('_')[-1])
    center = get_center(snap_id)
    sp = ds.sphere(center, (25.0, "kpc"))
    return sp.quantities.total_quantity(('gas', 'epsilon_gamma_incell')).to_value('erg/s')








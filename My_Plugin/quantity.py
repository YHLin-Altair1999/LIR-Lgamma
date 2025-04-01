import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yt
import astropy.units as u
import astropy.constants as c
from scipy.integrate import simpson
from .LoadData import get_center, get_snap_path
from .Add_Fields import add_fields
from My_Plugin.skirt.convert import get_units
import astropy.units as u
import astropy.constants as c
import h5py
from glob import glob
import os
from tqdm import tqdm

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

def L_gamma_yt(ds, center, aperture):
    ds = add_fields(ds)
    snap_id = int(ds.filename.split('/')[-1].split('.')[0].split('_')[-1])
    sp = ds.sphere(center, (aperture.value, aperture.unit))
    L_gamma = sp.quantities.total_quantity(('gas', 'epsilon_gamma_incell')).to_value('erg/s')*u.erg/u.s
    print('Gamma ray luminosity is', L_gamma)
    return L_gamma

def L_gamma_YHLin(galaxy, snap, aperture=25*u.kpc):
    print('Calculating gamma ray luminosity using custom built functions...')
    fnames = glob(os.path.join(get_snap_path(galaxy, snap), '*.hdf5'))
    print('The files are', fnames)
    fs = [h5py.File(fname, 'r') for fname in fnames]
    center = get_center(galaxy, snap)
    L_gamma = sum([L_gamma_YHLin_onefile(f, center, aperture) for f in fs]).to('erg/s')
    print(f'The total gamma ray luminosity is {L_gamma}')
    return L_gamma

def L_gamma_YHLin_onefile(f, center, aperture=25*u.kpc):
    '''
    GeV Gamma ray emissivity in units of erg/(s*cm**3).
    Ref: TK Chan et al. (2019) Eq. 6, 8
    https://arxiv.org/abs/1812.10496
    '''

    units = get_units(f)
    code_mass = units[0]
    code_length = units[1]
    code_velocity = units[2]

    r = np.linalg.norm(f['PartType0']['Coordinates'][:,:]*code_length - np.array(center)*code_length, axis=1)
    print(f'About {np.sum(r < aperture)/r.shape[0]*100:.2f}% of the particles are in the {aperture} aperture')

    beta_pi = 0.7
    x_e = f['PartType0']['ElectronAbundance'][:]
    f_Hydrogen = 1 - f['PartType0']['Metallicity'][:,1] - f['PartType0']['Metallicity'][:,0]
    E_cr = f['PartType0']['CosmicRayEnergy'] * code_mass * code_velocity**2
    density = f['PartType0']['Density'] * code_mass / code_length**3
    mass = f['PartType0']['Masses'] * code_mass
    volume = mass / density
    
    #n_n =  density * f_Hydrogen / c.m_p # thermal proton number density
    n_n =  density / c.m_p # thermal nucleon number density
    e_cr = E_cr / volume # CR energy density
    
    #Gamma_cr_had = 5.8e-16 * (1+0.28*x_e) * e_cr.to('erg*cm**-3').value * n_n.to('cm**-3').value * u.erg/(u.cm**3*u.s)
    Gamma_cr_had = 5.8e-16 * e_cr.to('erg*cm**-3').value * n_n.to('cm**-3').value * u.erg/(u.cm**3*u.s) # equ 6
    dL_gamma = 1/3 * beta_pi * Gamma_cr_had * volume # equ 8
    L_gamma_onefile = np.sum(dL_gamma[r < aperture])
    return L_gamma_onefile.to('erg/s')

def L_gamma_make_one_profile(galaxy: str, snap: int, rs):
    '''
    make the gamma ray luminosity profile of a given galaxy snapshot
    '''
    
    profile = np.zeros((rs.shape[0], 2))
    profile[:,0] = rs.to('cm').value

    print('Calculating gamma ray luminosity using custom built functions...')
    fnames = glob(os.path.join(get_snap_path(galaxy, snap), '*.hdf5'))
    print('The files are', fnames)
    fs = [h5py.File(fname, 'r') for fname in fnames]
    center = get_center(galaxy, snap)
    
    for f in fs:
        units = get_units(f)
        code_mass = units[0]
        code_length = units[1]
        code_velocity = units[2]

        r = np.linalg.norm(f['PartType0']['Coordinates'][:,:]*code_length - np.array(center)*code_length, axis=1)
        beta_pi = 0.7
        x_e = f['PartType0']['ElectronAbundance'][:]
        f_Hydrogen = 1 - f['PartType0']['Metallicity'][:,1] - f['PartType0']['Metallicity'][:,0]
        E_cr = f['PartType0']['CosmicRayEnergy'] * code_mass * code_velocity**2
        density = f['PartType0']['Density'] * code_mass / code_length**3
        mass = f['PartType0']['Masses'] * code_mass
        volume = mass / density
        
        n_n =  density / c.m_p # thermal nucleon number density
        e_cr = E_cr / volume # CR energy density
        
        Gamma_cr_had = 5.8e-16 * e_cr.to('erg*cm**-3').value * n_n.to('cm**-3').value * u.erg/(u.cm**3*u.s) # equ 6
        dL_gamma = 1/3 * beta_pi * Gamma_cr_had * volume # equ 8

        for i in tqdm(range(1, rs.shape[0])):
            profile[i,1] += np.sum(dL_gamma[(r > rs[i-1])*(r < rs[i])]).to('erg/s').value

    target_folder = '/tscc/lustre/ddn/scratch/yel051/tables/Lgamma_profiles'
    fname = os.path.join(target_folder, f'Lgamma_profile_{galaxy}_snap{snap:03d}.npy')
    np.save(fname, profile)
    return



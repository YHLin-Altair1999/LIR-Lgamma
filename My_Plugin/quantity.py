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
import h5py
from glob import glob
import os
from tqdm import tqdm
import scipy as sp
from astropy.cosmology import FlatLambdaCDM

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
    wav_integrate = np.logspace(np.log10(band[0].to('micron').value), np.log10(band[1].to('micron').value), 100)*u.micron
    L_nu_integrate = np.interp(wav_integrate.value, wav.value, L_nu.value)*L_nu.unit
    nu = c.c/wav_integrate
    L_IR = -simpson(L_nu_integrate.to('erg/(s*Hz)').value, x=nu.to('Hz').value)*u.erg/u.s
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

def get_C_p(e_cr, alpha_p, q):
    """Calculate C_p for CR protons"""

    rest_energy = c.m_p * c.c**2
    energy_integral = 0.5 * \
        sp.special.betainc((alpha_p-2)/2, (3-alpha_p)/2, 1/(1+q**2)) * \
        sp.special.beta((alpha_p-2)/2, (3-alpha_p)/2) + \
        q**(alpha_p-1) * (np.sqrt(1+q**2) - 1)
    C_p = e_cr * (alpha_p - 1) / (rest_energy * energy_integral)
    return C_p

def L_gamma_make_one_profile_Pfrommer(
        galaxy: str, snap: int, rs, 
        alpha_p=2.2, E_1=1*u.GeV, E_2=1000*u.GeV, q=0.5
        ):
    '''
    make the gamma ray luminosity profile of a given galaxy snapshot
    '''
    
    profile = np.zeros((rs.shape[0], 2))
    profile[:,0] = rs.to('cm').value

    print('Calculating gamma ray luminosity using custom built functions following Pfrommer et al. (2017)...')
    fnames = glob(os.path.join(get_snap_path(galaxy, snap), '*.hdf5'))
    print('The files are', fnames)
    fs = [h5py.File(fname, 'r') for fname in fnames]
    center = get_center(galaxy, snap)

    m_pi = 134.9768*u.MeV/c.c**2
    T_CMB = 2.726*u.K
    u_CMB = 0.260*u.eV/u.cm**3
    r_e = 2.8179403205e-15*u.m
    E_range = np.logspace(np.log10(E_1.to('GeV').value), np.log10(E_2.to('GeV').value), 10)*u.GeV
    
    for f in fs:
        units = get_units(f)
        code_mass = units[0]
        code_length = units[1]
        code_velocity = units[2]

        r = np.linalg.norm(f['PartType0']['Coordinates'][:,:]*code_length - np.array(center)*code_length, axis=1)
        E_cr = f['PartType0']['CosmicRayEnergy'] * code_mass * code_velocity**2
        density = f['PartType0']['Density'] * code_mass / code_length**3
        mass = f['PartType0']['Masses'] * code_mass
        B_vec = f['PartType0']['MagneticField'] * u.Gauss
        B = np.linalg.norm(B_vec, axis=1)
        u_B = B.to('Gauss').value**2/(8*np.pi)*u.erg/u.cm**3
        volume = mass / density
        
        n_n =  density / c.m_p # thermal nucleon number density
        e_cr = E_cr / volume # CR energy density

        alpha_e = alpha_p + 1
        alpha_nu = (alpha_e - 1)/2
        sigma_pp = 32*(0.96 + np.exp(4.4-2.4*alpha_p))*u.mbarn
        
        C_p = get_C_p(e_cr, alpha_p=alpha_p, q=q)
        
        delta = 0.14*alpha_p**(-1.6) + 0.44
        sigma_pp = 32*(0.96 + np.exp(4.4 - 2.4*alpha_p))*u.mbarn
        
        # Calculate s_pi: pion-decay gamma-ray luminosity
        energy_ratio = 2*E_range/(m_pi*c.c**2)
        energy_term = (energy_ratio**delta + energy_ratio**(-delta))**(-alpha_p/delta)
        mass_ratio = (c.m_p/(2*m_pi))**alpha_p
        interaction_factor = sigma_pp*n_n/(c.m_p*c.c)
        normalization = 16*C_p/(3*alpha_p)

        print(energy_ratio.shape, energy_term.shape, interaction_factor.shape, normalization.shape)
        # Calculate pion-decay gamma-ray luminosity
        s_pi = normalization[:, np.newaxis] * interaction_factor[:, np.newaxis] * mass_ratio * energy_term

        C_e = 16**(2-alpha_e) * sigma_pp * n_n * C_p * c.m_e*c.c**2 / \
              ((alpha_e-2)*c.sigma_T*(u_B+u_CMB)) * (c.m_p/c.m_e)**(alpha_e-2)
               
        f_IC = 2**(alpha_e+3)*(alpha_e**2 + 4*alpha_e + 11)/ \
               ((alpha_e+3)**2*(alpha_e+5)*(alpha_e+1)) * sp.special.gamma((alpha_e+5)/2) * sp.special.zeta((alpha_e+5)/2)
        
        s_IC = C_e[:, np.newaxis]*8*np.pi**2*r_e**2/(c.h**3*c.c**2) * \
            (c.k_B*T_CMB)**(3+alpha_nu) * f_IC * E_range**(-alpha_nu-1)

        s_tot = s_pi + s_IC
        Lambda_total = sp.integrate.simpson(
            (s_tot * E_range).to(u.s**-1*u.cm**-3).value, x=E_range.to('GeV').value
            ) * u.GeV*u.s**-1*u.cm**-3
        dL_gamma = Lambda_total * volume # equ 8

        for i in tqdm(range(1, rs.shape[0])):
            profile[i,1] += np.sum(dL_gamma[(r > rs[i-1])*(r < rs[i])]).to('erg/s').value

    target_folder = '/tscc/lustre/ddn/scratch/yel051/tables/Lgamma_profiles'
    fname = os.path.join(target_folder, f'Lgamma_profile_{galaxy}_snap{snap:03d}.npy')
    np.save(fname, profile)
    return

def SFR_make_one_profile(
        galaxy: str, snap: int, rs, 
        ):
    '''
    make the gamma ray luminosity profile of a given galaxy snapshot
    '''
    
    profile = np.zeros((rs.shape[0], 2))
    profile[:,0] = rs.to('cm').value

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
        SFR = f['PartType0']['StarFormationRate']*u.M_sun/u.yr

        for i in tqdm(range(1, rs.shape[0])):
            profile[i,1] += np.sum(SFR[(r > rs[i-1])*(r < rs[i])]).to('M_sun/yr').value

    target_folder = '/tscc/lustre/ddn/scratch/yel051/tables/SFR_profiles'
    fname = os.path.join(target_folder, f'SFR_profile_{galaxy}_snap{snap:03d}.npy')
    np.save(fname, profile)
    return

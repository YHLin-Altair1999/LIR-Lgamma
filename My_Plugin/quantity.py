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
from gizmo_analysis import gizmo_star

def L_IR(fname, wav_min=8*u.micron, wav_max=1000*u.micron):
    '''
    Calculate the total IR luminosity from SKIRT-generated SED files.
    Automatically extracts column information, units, and distance from file header.
    
    Parameters:
    -----------
    fname : str
        Path to the SKIRT SED file
    band : astropy.units.Quantity
        Wavelength range for integration, default is [8, 1000] micron
        
    Returns:
    --------
    L_IR : astropy.units.Quantity
        Total infrared luminosity in erg/s
    '''
    # Read header to extract metadata
    header_lines = []
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                header_lines.append(line.strip())
                if i >= 15:  # Assume header is no more than 15 lines
                    print('Header seems a bit too long, might want to check.')
            else:
                break
    
    # Extract distance information from header
    distance_line = [line for line in header_lines if "distance" in line][0]
    distance_value = float(distance_line.split()[-2])
    distance_unit = distance_line.split()[-1]
    d = distance_value * u.Unit(distance_unit)
    
    # Extract column names from header
    column_info = [line for line in header_lines if "column" in line]
    names = ['wavelength']
    for line in column_info[1:]:  # First column is wavelength
        name = line.split(':')[1].split(';')[0].strip()
        names.append(name)  # Extract the first word (e.g., "total" from "total flux")
    
    # Read data with extracted column names
    df = pd.read_csv(fname, names=names, comment='#', sep=' ')
    
    # Calculate luminosity
    F_nu = np.array(df['total flux'])*u.Jy
    L_nu = 4*np.pi*d**2*F_nu
    wav = np.array(df['wavelength'])*u.micron
    wav_integrate = np.logspace(np.log10(wav_min.value), np.log10(wav_max.value), 100)*wav_min.unit
    L_nu_integrate = np.interp(wav_integrate.value, wav.value, L_nu.value)*L_nu.unit
    nu = c.c/wav_integrate
    L_IR = -simpson(L_nu_integrate.value, x=nu.value) * (L_nu_integrate.unit * nu.unit)
    return L_IR.to('erg/s')

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

        #print(energy_ratio.shape, energy_term.shape, interaction_factor.shape, normalization.shape)
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
        galaxy: str, snap: int, rs, sfr_type='FIR'
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

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        
        def _get_sfr_from_stars(f, center, code_mass, code_length, tau_sfr=10*u.Myr):
            cosmo = FlatLambdaCDM(
                    H0 = f['Header'].attrs.get('HubbleParam')*100*u.km/(u.s*u.Mpc),
                    Om0 = f['Header'].attrs.get('Omega0')
                    )
            z_now = f['Header'].attrs.get('Redshift')
            t_now = cosmo.lookback_time(z_now)
            a_sf  = np.array(f['PartType4']['StellarFormationTime'])
            z_sf  = 1/a_sf - 1
            t_sf  = cosmo.lookback_time(z_sf)
            age   = (t_sf - t_now).to('Gyr')
            mass = np.array(f['PartType4']['Masses'])*code_mass
            metallicity = np.array(f['PartType4']['Metallicity'])[:,0]
            
            mask = age > tau_sfr

            mass_loss = gizmo_star.MassLossClass('fire2')
            mass_loss_frac = mass_loss.get_mass_loss_from_spline(
                                                age.to('Gyr').value, # note that age is defined in Gyr
                                                metallicities=metallicity/0.02
                                                )
            mass_formed = mass / (1 - mass_loss_frac)
            mass_formed[mask] = 0
            SFR = mass_formed / tau_sfr
            return SFR

        match sfr_type:
            case 'gas':
                r = np.linalg.norm(f['PartType0']['Coordinates'][:,:]*code_length - np.array(center)*code_length, axis=1)
                SFR = f['PartType0']['StarFormationRate']*u.M_sun/u.yr
            case 'Ha':
                r = np.linalg.norm(f['PartType4']['Coordinates'][:,:]*code_length - np.array(center)*code_length, axis=1)
                SFR = _get_sfr_from_stars(f, center, code_mass, code_length, tau_sfr=2*u.Myr)
            case 'FIR':
                r = np.linalg.norm(f['PartType4']['Coordinates'][:,:]*code_length - np.array(center)*code_length, axis=1)
                SFR = _get_sfr_from_stars(f, center, code_mass, code_length, tau_sfr=100*u.Myr)

        for i in tqdm(range(1, rs.shape[0])):
            profile[i,1] += np.sum(SFR[(r > rs[i-1])*(r < rs[i])]).to('M_sun/yr').value

    target_folder = '/tscc/lustre/ddn/scratch/yel051/tables/SFR_profiles'
    fname = os.path.join(target_folder, f'SFR_profile_{galaxy}_snap{snap:03d}.npy')
    np.save(fname, profile)
    return


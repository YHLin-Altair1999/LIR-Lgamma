import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import logging
import os
from glob import glob
from gizmo_analysis import gizmo_star
from astropy.cosmology import FlatLambdaCDM
from My_Plugin.LoadData import get_center, get_angular_momentum, get_snap_path, get_radius
from My_Plugin.skirt.dust_scaling import dust_to_gas_ratio_RemyRuyer, dust_to_gas_ratio_Galliano
from My_Plugin.general_util import get_units, get_data, align_axis
logging.basicConfig(level=logging.INFO)
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    'font.family': 'serif'
    })

default_figsize = (5, 3)

def convert_gas(
        galaxy, snap_id=600, rotate=True, r_max=30*u.kpc
        ):
    logging.info('Converting gas particles...')
    fs = get_data(galaxy, snap_id)
    code_length = get_units(fs[0])[1]
    center = get_center(galaxy, snap_id)*code_length
    output = np.concatenate([convert_gas_onefile(f, center, r_max) for f in fs], axis=0)
    print('The total dust mass is', np.sum(output[:,5]*output[:,4]), 'M_sun')
    print('The averaged dust to gas ratio is', np.sum(output[:,5]*output[:,4]) / np.sum(output[:,4]))
    if rotate:
        output = align_axis(galaxy, snap_id, output)
    header = open('/tscc/lustre/ddn/scratch/yel051/My_Plugin/skirt/skirt_header_gas.txt', 'r').read()
    np.savetxt("gas.txt", output, delimiter=" ", header=header)

    fig, axes = plt.subplots(ncols=3, figsize=(12,4))
    slice_part_xy = output[np.abs(output[:,2])<output[:,3]]
    slice_part_yz = output[np.abs(output[:,0])<output[:,3]]
    slice_part_xz = output[np.abs(output[:,1])<output[:,3]]
    axes[0].scatter(slice_part_xy[:,0], slice_part_xy[:,1], s=0.1)
    axes[1].scatter(slice_part_yz[:,1], slice_part_yz[:,2], s=0.1)
    axes[2].scatter(slice_part_xz[:,0], slice_part_xz[:,2], s=0.1)
    for ax in axes:
        ax.set_xlim(-2.5e4,2.5e4)
        ax.set_ylim(-2.5e4,2.5e4)
        ax.set_aspect('equal')
    fig.savefig('gas_particle_slice.png', )
    return

def convert_gas_onefile(
        f, 
        center=np.array([0,0,0])*u.kpc,
        r_max = 30*u.kpc
        ):

    units = get_units(f)
    code_mass = units[0]
    code_length = units[1]
    code_velocity = units[2]

    # Set up cosmology
    cosmo = FlatLambdaCDM(
        H0 = f['Header'].attrs.get('HubbleParam')*100*u.km/(u.s*u.Mpc),
        Om0 = f['Header'].attrs.get('Omega0')
        )

    # Column 1-3: coordinates
    coords = np.array(f['PartType0']['Coordinates'])*code_length
    coords -= center
    r = np.linalg.norm(coords.to('kpc').value, axis=1)*u.kpc
    
    # Column 4: smoothing length (pc)
    l_smooth = np.array(f['PartType0']['SmoothingLength'])*code_length
    
    # Column 5: initial mass (Msun)
    mass = np.array(f['PartType0']['Masses'])*code_mass
    
    # Column 6: metallicity (1)
    '''
    Note: we are only using metallicity 00
    '''
    metallicity = np.array(f['PartType0']['Metallicity'])[:,0]
    He_mass_fraction = np.array(f['PartType0']['Metallicity'])[:,1]
    H_mass_fraction = 1 - metallicity - He_mass_fraction
    Oxygen_number_fraction = np.array(f['PartType0']['Metallicity'])[:,4] / 16
    Hydrogen_number_fraction = H_mass_fraction / 1
    nO_nH_ratio = Oxygen_number_fraction / Hydrogen_number_fraction
    logOH_12 = np.log10(nO_nH_ratio) + 12

    dust_to_gas_ratio = dust_to_gas_ratio_RemyRuyer(nO_nH_ratio)
    #dust_to_gas_ratio /= 0.02/0.013
    #dust_to_gas_ratio = dust_to_gas_ratio_Galliano(nO_nH_ratio)
    dust_to_metal_ratio = dust_to_gas_ratio / metallicity

    # Put a cap on the dust-to-metal ratio at 0.5
    dust_to_gas_ratio[dust_to_metal_ratio > 0.5] = 0.5 * metallicity[dust_to_metal_ratio > 0.5]
    dust_to_metal_ratio = dust_to_gas_ratio / metallicity

    mask = r < r_max

    fig, ax = plt.subplots(figsize=default_figsize)
    hist, bin_edges = np.histogram(dust_to_gas_ratio[mask], bins=50, range=(1e-4, 1e-1), weights=mass.to('M_sun').value[mask])
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.bar(bin_centers, hist, width=bin_edges[1:]-bin_edges[:-1], alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r'$M_{\rm d}/M_{\rm gas}$ ratio')
    ax.set_ylabel(r'$M_{\rm gas}$')
    plt.tight_layout()
    plt.savefig('dust_to_gas_ratio.png', dpi=300)

    fig, ax = plt.subplots(figsize=default_figsize)
    hist, bin_edges = np.histogram(metallicity[mask], bins=50, range=(1e-4, 1e-1), weights=mass.to('M_sun').value[mask])
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.bar(bin_centers, hist, width=bin_edges[1:]-bin_edges[:-1], alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r'Metallicity ($Z$)')
    ax.set_ylabel(r'$M_{\rm gas}$')
    plt.tight_layout()
    plt.savefig('metallicity_total.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=default_figsize)
    hist, bin_edges = np.histogram(logOH_12[mask], bins=50, range=(7, 10), weights=mass.to('M_sun').value[mask])
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.bar(bin_centers, hist, width=bin_edges[1:]-bin_edges[:-1], alpha=0.7)
    ax.set_xlabel(r'$\log_{10}(n_{\rm O}/n_{\rm H}) + 12$')
    ax.set_ylabel(r'$M_{\rm gas}$')
    plt.tight_layout()
    plt.savefig('metallicity_Oxygen.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=default_figsize)
    hist, bin_edges = np.histogram(dust_to_metal_ratio[mask], bins=50, range=(1e-2, 1e0), weights=mass.to('M_sun').value[mask])
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.bar(bin_centers, hist, width=bin_edges[1:]-bin_edges[:-1], alpha=0.7)
    ax.set_xlabel(r'$M_{\rm d}/M_{Z}$')
    ax.set_ylabel(r'$M_{\rm gas}$')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('dust_to_metal_ratio.png', dpi=300)
    plt.close()
    # Column 7: temperature
    '''
    See http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps-reading
    Search "convert internal energy to TEMPERATURE"
    '''
    e_int = np.array(f['PartType0']['InternalEnergy'])*code_velocity**2
    helium_mass_fraction = np.array(f['PartType0']['Metallicity'])[:,1]
    y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
    e_abundance = np.array(f['PartType0']['ElectronAbundance'])
    mu = (1 + 4*y_helium) / (1 + y_helium + e_abundance)
    mean_molecular_weight = mu*c.m_p
    gamma = 5/3
    temperature = mean_molecular_weight * (gamma-1) * e_int / c.k_B
    
    output = np.zeros((mass.shape[0], 7))
    output[:,:3] = coords.to('pc').value
    output[:,3] = l_smooth.to('pc').value
    output[:,4] = mass.to('M_sun').value
    #output[:,5] = metallicity
    output[:,5] = dust_to_gas_ratio
    output[:,6] = temperature.to('K').value
    output = output[r<r_max]

    return output

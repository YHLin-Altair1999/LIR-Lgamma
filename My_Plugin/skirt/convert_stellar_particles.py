import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import os
import logging

from glob import glob
from astropy.cosmology import FlatLambdaCDM
from My_Plugin.enclose import find_minimal_enclosing_radius_kdtree
from My_Plugin.LoadData import get_center, get_angular_momentum, get_snap_path, get_radius
from My_Plugin.general_util import get_units, get_data, align_axis
from gizmo_analysis import gizmo_star
logging.basicConfig(level=logging.INFO)
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    'font.family': 'serif'
    })
default_figsize = (5, 3)

def convert_stellar(
        galaxy, snap_id=600, rotate=True, r_max=30*u.kpc
        ):
    
    logging.info('Converting stellar particles...')
    fs = get_data(galaxy, snap_id)
    code_length = get_units(fs[0])[1]
    center = get_center(galaxy, snap_id)*code_length
    output = np.concatenate([convert_stellar_onefile(f, center, r_max) for f in fs], axis=0)
    if rotate:
        output = align_axis(galaxy, snap_id, output)
    
    # Calculate the stellar smoothing length
    N_enclose = 64
    logging.info(f'Smoothing length calculated using KDTree with {N_enclose} enclosed particles')
    l_smooth = find_minimal_enclosing_radius_kdtree(output[:,:3], N_enclose)*u.pc
    output[:,3] = l_smooth.to('pc').value

    # Calculate the mass loss rate
    '''
    To know the initial mass for importing into STARBURST99, we need to include the mass loss model.
    Moreover, the mass loss model for FIRE-2 and FIRE-3 are different.
    To use this, we import the model by Andrew Wetzel:
    https://bitbucket.org/awetzel/gizmo_analysis/src/master/
    '''
    mass_loss = gizmo_star.MassLossClass('fire2')
    mass_loss_frac = mass_loss.get_mass_loss_from_spline(
                                        output[:,6]*1000, # note that age is defined in Gyr
                                        metallicities=output[:,5])
    output[:,4] /= (1 - mass_loss_frac)
    
    # Visualize it for inspection
    fig, ax = plt.subplots(figsize=default_figsize)
    counts, bins = np.histogram(mass_loss_frac)
    ax.stairs(counts, bins)
    ax.set_xlabel('mass loss fraction')
    ax.set_ylabel('Number of particles')
    plt.tight_layout()
    fig.savefig('mass_loss_frac.png', dpi=300)
    plt.close()

    header = open('/tscc/lustre/ddn/scratch/yel051/My_Plugin/skirt/skirt_header_stars.txt', 'r').read()
    np.savetxt("stars.txt", output, delimiter=" ", header=header)

    fig, axes = plt.subplots(ncols=3, figsize=(9,3), sharey=True)
    plt.subplots_adjust(left=0.08, right=0.85, wspace=0.)
    slice_part_xy = output[np.abs(output[:,2])<output[:,3]]
    slice_part_yz = output[np.abs(output[:,0])<output[:,3]]
    slice_part_xz = output[np.abs(output[:,1])<output[:,3]]
    
    cmap = plt.get_cmap('Spectral_r')
    age_min = 1e-1 # Gyr
    age_max = 1e1 # Gyr
    xyplane = axes[0].scatter(
        slice_part_xy[:,0], slice_part_xy[:,1], 
        s=1, alpha=0.1, edgecolor='None',
        c=cmap(np.log10(output[:,6][np.abs(output[:,2])<output[:,3]]))
        )
    yzplane = axes[1].scatter(
        slice_part_yz[:,1], slice_part_yz[:,2], 
        s=1, alpha=0.1, edgecolor='None', 
        c=cmap(np.log10(output[:,6][np.abs(output[:,0])<output[:,3]]))
        )
    xzplane = axes[2].scatter(
        slice_part_xz[:,0], slice_part_xz[:,2], 
        s=1, alpha=0.1, edgecolor='None', 
        c=cmap(np.log10(output[:,6][np.abs(output[:,1])<output[:,3]]))
        )
    norm = plt.Normalize(vmin=np.log10(age_min), vmax=np.log10(age_max)) 
    cbar_ax = fig.add_axes([0.87, 0.11, 0.02, 0.78])  # Adjust position as needed
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('log(Age/Gyr)')  # Set a label for the colorbar
    box_size = r_max.to('pc').value
    for ax in axes:
        ax.set_xlim(-box_size/2,box_size/2)
        ax.set_ylim(-box_size/2,box_size/2)
        ax.set_aspect('equal')
    fig.savefig('star_particle_slice.png', dpi=300)
    plt.close()
    return

def convert_stellar_onefile(
        f,
        center=np.array([0,0,0])*u.kpc,
        r_max = 30*u.kpc
        ):
    units = get_units(f)
    code_mass = units[0]
    code_length = units[1]
    # Set up cosmology
    cosmo = FlatLambdaCDM(
        H0 = f['Header'].attrs.get('HubbleParam')*100*u.km/(u.s*u.Mpc),
        Om0 = f['Header'].attrs.get('Omega0')
        )

    # Column 1-3: coordinates
    coords = np.array(f['PartType4']['Coordinates'])*code_length
    coords -= center
    r = np.linalg.norm(coords, axis=1)
    
    # Column 4: smoothing length
    # This part is handled after combining all the hdf5 files
    
    # Column 5: initial mass (Msun)
    '''
    In FIRE snapshots the mass of the stellar particles are their current mass.
    We will adjust for the mass loss a bit later in the convert_stellar function.
    '''
    mass = np.array(f['PartType4']['Masses'])*code_mass
    
    # Column 6: metallicity (1)
    '''
    Note: we are only using metallicity 00
    '''
    metallicity = np.array(f['PartType4']['Metallicity'])[:,0]
    
    # Column 7: stellar age
    z_now = f['Header'].attrs.get('Redshift')
    t_now = cosmo.lookback_time(z_now)
    a_sf  = np.array(f['PartType4']['StellarFormationTime'])
    z_sf  = 1/a_sf-1
    t_sf  = cosmo.lookback_time(z_sf)
    age   = (t_sf - t_now).to('Gyr').value

    # Adjust for mass loss

    output = np.zeros((mass.shape[0], 7))
    output[:,:3] = coords.to('pc').value
    output[:,4] = mass.to('M_sun').value
    output[:,5] = metallicity
    output[:,6] = age
    output = output[(r<r_max)]

    return output

def draw_SFH(array: np.ndarray):
    '''
    input: array
    1st column: stellar mass in Msun
    2nd column: stellar age in Gyr
    '''
    fig, ax = plt.subplots(figsize=default_figsize)

    return None



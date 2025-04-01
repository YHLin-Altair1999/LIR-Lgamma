import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as c
from ..enclose import find_minimal_enclosing_radius_kdtree
from ..LoadData import get_center, get_angular_momentum, get_snap_path
from glob import glob
import logging
from gizmo_analysis import gizmo_star
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    code_mass = 1e10*u.M_sun
    code_length = 1e0*u.kpc*a
    code_velocity = 1*u.km/u.s*np.sqrt(a)
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

def convert_stellar(galaxy, snap_id=600, rotate=True, r_max=30*u.kpc):
    
    logging.info('Converting stellar particles...')
    fs = get_data(galaxy, snap_id)
    units = get_units(fs[0])
    code_mass = units[0]
    code_length = units[1]
    code_velocity = units[2]
    center = get_center(galaxy, snap_id)*code_length
    output = np.concatenate([convert_stellar_onefile(f, center, r_max) for f in fs], axis=0)
    if rotate:
        output = align_axis(galaxy, snap_id, output)
    
    # Calculate the stellar smoothing length
    N_enclose = 32
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
                                        metallicities=output[:,5]/0.02)
    output[:,4] /= (1 - mass_loss_frac)
    
    # Visualize it for inspection
    fig, ax = plt.subplots()
    counts, bins = np.histogram(mass_loss_frac)
    ax.stairs(counts, bins)
    ax.set_xlabel('mass loss fraction')
    ax.set_ylabel('Number of particles')
    fig.savefig('mass_loss_frac.png', dpi=300)
    plt.close()

    header = open('/tscc/lustre/ddn/scratch/yel051/My_Plugin/skirt/skirt_header_stars.txt', 'r').read()
    np.savetxt("stars.txt", output, delimiter=" ", header=header)

    # Caution! Removing old stars for investigating issue with m12i_sc
    #output = output[output[:,6]<0.1]

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
    box_size = 2.5e4
    for ax in axes:
        ax.set_xlim(-box_size/2,box_size/2)
        ax.set_ylim(-box_size/2,box_size/2)
        ax.set_aspect('equal')
    fig.savefig('star_particle_slice.jpg', dpi=300)
    plt.close()
    return

def convert_stellar_onefile(f,
                            center=np.array([0,0,0])*u.kpc,
                            r_max = 30*u.kpc):
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
    coords = np.array(f['PartType4']['Coordinates'])*code_length
    coords -= center
    r = np.linalg.norm(coords, axis=1)
    
    # Column 4: smoothing length
    # This part is handled after combining all the hdf5 files
    
    # Column 5: initial mass (Msun)
    '''
    In FIRE snapshots the mass of the stellar particles are their current mass.
    We will adjust for the mass loss a bit later.
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

def convert_gas(galaxy, snap_id=600, rotate=True, r_max=30*u.kpc):
    logging.info('Converting gas particles...')
    fs = get_data(galaxy, snap_id)
    units = get_units(fs[0])
    code_mass = units[0]
    code_length = units[1]
    code_velocity = units[2]
    center = get_center(galaxy, snap_id)*code_length
    output = np.concatenate([convert_gas_onefile(f, center, r_max) for f in fs], axis=0)
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

def convert_gas_onefile(f, 
                        center=np.array([0,0,0])*u.kpc,
                        r_max = 30*u.kpc):

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
    '''
    Question: does stellar mass in FIRE consider stellar mass loss?
    '''
    mass = np.array(f['PartType0']['Masses'])*code_mass
    
    # Column 6: metallicity (1)
    '''
    Note: we are only using metallicity 00
    '''
    metallicity = np.array(f['PartType0']['Metallicity'])[:,0]
    
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
    output[:,5] = metallicity
    output[:,6] = temperature.to('K').value
    output = output[r<r_max]

    return output

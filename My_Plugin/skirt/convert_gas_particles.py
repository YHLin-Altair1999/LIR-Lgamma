import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import logging
import os
from glob import glob
from gizmo_analysis import gizmo_star
from My_Plugin.LoadData import get_center, get_angular_momentum, get_snap_path, get_radius
from My_Plugin.skirt.dust_scaling import dust_to_gas_ratio_RemyRuyer, dust_to_gas_ratio_Galliano, dust_to_gas_ratio_RemyRuyer_nodrop
from My_Plugin.general_util import get_units, get_data, align_axis, get_cosmology
from matplotlib.colors import LogNorm
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
    
    # Process all files and collect results
    outputs = []
    stats_data = []
    for f in fs:
        output, stats = convert_gas_onefile(f, center, r_max)
        outputs.append(output)
        stats_data.append(stats)
    
    # Concatenate all outputs
    output = np.concatenate(outputs, axis=0)
    print('The total dust mass is', np.sum(output[:,5]*output[:,4]), 'M_sun')
    print('The averaged dust to gas ratio is', np.sum(output[:,5]*output[:,4]) / np.sum(output[:,4]))
    
    if rotate:
        output = align_axis(galaxy, snap_id, output)
    header = open('/tscc/lustre/ddn/scratch/yel051/My_Plugin/skirt/skirt_header_gas.txt', 'r').read()
    np.savetxt("gas.txt", output, delimiter=" ", header=header)

    # Save intermediate statistical data
    np.save('gas_statistics.npy', stats_data)
    logging.info(f'Saved statistical data from {len(stats_data)} files to gas_statistics.npy')
    
    # Generate plots with aggregated data
    summary_stats = generate_statistical_plots(stats_data)
    
    # Create particle slice plot
    fig, axes = plt.subplots(ncols=3, figsize=(12,4))
    slice_part_xy = output[np.abs(output[:,2])<output[:,3]]
    slice_part_yz = output[np.abs(output[:,0])<output[:,3]]
    slice_part_xz = output[np.abs(output[:,1])<output[:,3]]
    axes[0].scatter(slice_part_xy[:,0], slice_part_xy[:,1], s=0.1)
    axes[1].scatter(slice_part_yz[:,1], slice_part_yz[:,2], s=0.1)
    axes[2].scatter(slice_part_xz[:,0], slice_part_xz[:,2], s=0.1)
    for ax in axes:
        ax.set_xlim(-r_max.to('pc').value/2,r_max.to('pc').value/2)
        ax.set_ylim(-r_max.to('pc').value/2,r_max.to('pc').value/2)
        ax.set_aspect('equal')
    fig.savefig('gas_particle_slice.png', )
    
    return summary_stats

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
    cosmo = get_cosmology(f)

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
    #dust_to_gas_ratio = dust_to_gas_ratio_RemyRuyer_nodrop(nO_nH_ratio)
    #dust_to_gas_ratio = dust_to_gas_ratio_Galliano(nO_nH_ratio)
    dust_to_metal_ratio = dust_to_gas_ratio / metallicity

    # Put a cap on the dust-to-metal ratio at 0.5
    # dust_to_gas_ratio[dust_to_metal_ratio > 0.5] = 0.5 * metallicity[dust_to_metal_ratio > 0.5]
    
    # Only for test!!!!!!
    # dust_to_gas_ratio = 0.5 * metallicity
    # dust_to_metal_ratio = dust_to_gas_ratio / metallicity

    mask = r < r_max

    # Compute histograms and statistics instead of storing raw data
    mass_values = mass.to('M_sun').value[mask]
    
    # Histogram 1: Dust-to-gas ratio
    hist_dtg, bin_edges_dtg = np.histogram(dust_to_gas_ratio[mask], bins=50, range=(1e-4, 1e-1), weights=mass_values)
    
    # Histogram 2: Metallicity
    hist_met, bin_edges_met = np.histogram(metallicity[mask], bins=50, range=(1e-4, 1e-1), weights=mass_values)
    
    # Histogram 3: Oxygen abundance
    hist_oh, bin_edges_oh = np.histogram(logOH_12[mask], bins=50, range=(7, 10), weights=mass_values)
    
    # Histogram 4: Dust-to-metal ratio
    hist_dtm, bin_edges_dtm = np.histogram(dust_to_metal_ratio[mask], bins=50, range=(1e-2, 1e0), weights=mass_values)
    
    # 2D histogram: Metallicity vs Oxygen abundance
    hist_2d, xedges_2d, yedges_2d = np.histogram2d(
        np.log10(metallicity)[mask], logOH_12[mask], bins=(150, 50), range=((-4, -1), (7, 10)), weights=mass_values
    )
    # Store only the computed histograms and bin edges
    stats = {
        'hist_dust_to_gas': hist_dtg,
        'bin_edges_dust_to_gas': bin_edges_dtg,
        'hist_metallicity': hist_met,
        'bin_edges_metallicity': bin_edges_met,
        'hist_oxygen': hist_oh,
        'bin_edges_oxygen': bin_edges_oh,
        'hist_dust_to_metal': hist_dtm,
        'bin_edges_dust_to_metal': bin_edges_dtm,
        'hist_2d_met_oh': hist_2d,
        'xedges_2d': xedges_2d,
        'yedges_2d': yedges_2d,
        'total_mass': np.sum(mass_values)
    }

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

    return output, stats

def generate_statistical_plots(stats_data):
    """Generate statistical plots from aggregated histogram data from all files"""
    
    # Initialize combined histograms
    combined_hist_dtg = None
    combined_hist_met = None
    combined_hist_oh = None
    combined_hist_dtm = None
    combined_hist_2d = None
    
    # Get bin edges from first file (assuming all files use same binning)
    bin_edges_dtg = stats_data[0]['bin_edges_dust_to_gas']
    bin_edges_met = stats_data[0]['bin_edges_metallicity']
    bin_edges_oh = stats_data[0]['bin_edges_oxygen']
    bin_edges_dtm = stats_data[0]['bin_edges_dust_to_metal']
    xedges_2d = stats_data[0]['xedges_2d']
    yedges_2d = stats_data[0]['yedges_2d']
    
    # Sum histograms from all files
    total_mass = 0
    for stats in stats_data:
        if combined_hist_dtg is None:
            combined_hist_dtg = stats['hist_dust_to_gas'].copy()
            combined_hist_met = stats['hist_metallicity'].copy()
            combined_hist_oh = stats['hist_oxygen'].copy()
            combined_hist_dtm = stats['hist_dust_to_metal'].copy()
            combined_hist_2d = stats['hist_2d_met_oh'].copy()
        else:
            combined_hist_dtg += stats['hist_dust_to_gas']
            combined_hist_met += stats['hist_metallicity']
            combined_hist_oh += stats['hist_oxygen']
            combined_hist_dtm += stats['hist_dust_to_metal']
            combined_hist_2d += stats['hist_2d_met_oh']
        
        total_mass += stats['total_mass']
    
    logging.info(f'Combined data from {len(stats_data)} files with total mass: {total_mass:.2e} M_sun')
    
    # Plot 1: Dust-to-gas ratio histogram
    fig, ax = plt.subplots(figsize=default_figsize)
    bin_centers_dtg = 0.5 * (bin_edges_dtg[1:] + bin_edges_dtg[:-1])
    ax.bar(bin_centers_dtg, combined_hist_dtg, width=bin_edges_dtg[1:]-bin_edges_dtg[:-1], alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r'$M_{\rm d}/M_{\rm gas}$ ratio')
    ax.set_ylabel(r'$M_{\rm gas}$ ($M_{\odot}$)')
    plt.tight_layout()
    plt.savefig('dust_to_gas_ratio_total.png', dpi=300)
    plt.close()

    # Plot 2: Metallicity histogram
    fig, ax = plt.subplots(figsize=default_figsize)
    bin_centers_met = 0.5 * (bin_edges_met[1:] + bin_edges_met[:-1])
    ax.bar(bin_centers_met, combined_hist_met, width=bin_edges_met[1:]-bin_edges_met[:-1], alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r'Metallicity ($Z$)')
    ax.set_ylabel(r'$M_{\rm gas}$ ($M_{\odot}$)')
    plt.tight_layout()
    plt.savefig('metallicity_total.png', dpi=300)
    plt.close()

    # Plot 3: Oxygen abundance histogram
    fig, ax = plt.subplots(figsize=default_figsize)
    bin_centers_oh = 0.5 * (bin_edges_oh[1:] + bin_edges_oh[:-1])
    ax.bar(bin_centers_oh, combined_hist_oh, width=bin_edges_oh[1:]-bin_edges_oh[:-1], alpha=0.7)
    ax.set_xlabel(r'$\log_{10}(n_{\rm O}/n_{\rm H}) + 12$')
    ax.set_ylabel(r'$M_{\rm gas}$ ($M_{\odot}$)')
    plt.tight_layout()
    plt.savefig('metallicity_Oxygen_total.png', dpi=300)
    plt.close()

    # Plot 4: Dust-to-metal ratio histogram
    fig, ax = plt.subplots(figsize=default_figsize)
    bin_centers_dtm = 0.5 * (bin_edges_dtm[1:] + bin_edges_dtm[:-1])
    ax.bar(bin_centers_dtm, combined_hist_dtm, width=bin_edges_dtm[1:]-bin_edges_dtm[:-1], alpha=0.7)
    ax.set_xlabel(r'$M_{\rm d}/M_{Z}$')
    ax.set_ylabel(r'$M_{\rm gas}$ ($M_{\odot}$)')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('dust_to_metal_ratio_total.png', dpi=300)
    plt.close()

    # Plot 5: 2D histogram of metallicity vs oxygen abundance
    fig, ax = plt.subplots(figsize=default_figsize)
    dynamic_range = 1e4
    combined_hist_2d[combined_hist_2d < np.max(combined_hist_2d)/dynamic_range] = np.max(combined_hist_2d)/dynamic_range
    im = ax.imshow(combined_hist_2d.T, origin='lower', aspect='auto',
                   extent=(xedges_2d[0], xedges_2d[-1], yedges_2d[0], yedges_2d[-1]),
                   cmap='inferno', norm=LogNorm(vmin=np.max(combined_hist_2d)/dynamic_range, vmax=np.max(combined_hist_2d))
                   )
    
    #ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.set_title('Total gas mass in 2D histogram')
    fig.colorbar(im, ax=ax, label=r'Total gas mass ($M_{\odot}$)')
    ax.set_ylabel(r'$\log_{10}({\rm O/H}) + 12$')
    ax.set_xlabel(r'$\log_{10}(Z)$')
    #ax.set_xlim(0.01, 0.03)
    ax.axvline(np.log10(0.02), color='#555555', linestyle='-', label='Solar metallicity (old)', alpha=0.5)
    ax.axvline(np.log10(0.014), color='#888888', linestyle='-', label='Solar metallicity (new)', alpha=0.5)
    ax.axhline(8.69, color='#999999', linestyle='-', label='Solar Oxygen abundance', alpha=0.5)
    ax.text(0.05, 0.9, str(os.getcwd()).split('/')[-2].replace('_', ' '), transform=ax.transAxes, fontsize=12, color='w', ha='left', va='top')
    #ax.legend()
    plt.tight_layout()
    plt.savefig('OH12_vs_metallicity.png', dpi=300)
    plt.close()
    
    logging.info('Generated aggregated statistical plots')
    
    return {
        'total_mass': total_mass,
        'n_files': len(stats_data),
        'combined_histograms': {
            'dust_to_gas': (combined_hist_dtg, bin_edges_dtg),
            'metallicity': (combined_hist_met, bin_edges_met),
            'oxygen': (combined_hist_oh, bin_edges_oh),
            'dust_to_metal': (combined_hist_dtm, bin_edges_dtm),
            '2d_met_oh': (combined_hist_2d, xedges_2d, yedges_2d)
        }
    }

def load_and_plot_statistics(stats_file='gas_statistics.npy'):
    """Load previously saved statistics and generate plots"""
    try:
        stats_data = np.load(stats_file, allow_pickle=True)
        summary_stats = generate_statistical_plots(stats_data)
        logging.info(f'Loaded statistics from {stats_file} and generated plots')
        return summary_stats
    except FileNotFoundError:
        logging.error(f'Statistics file {stats_file} not found. Run convert_gas first.')
    except Exception as e:
        logging.error(f'Error loading statistics: {e}')

def print_statistics_summary(stats_file='gas_statistics.npy'):
    """Print a summary of the saved statistics"""
    try:
        stats_data = np.load(stats_file, allow_pickle=True)
        
        total_mass = sum(stats['total_mass'] for stats in stats_data)
        
        print(f"Statistics summary from {len(stats_data)} files:")
        #print(f"File numbers: {sorted(file_numbers)}")
        print(f"Total gas mass: {total_mass:.2e} M_sun")
        print(f"Average mass per file: {total_mass/len(stats_data):.2e} M_sun")        
        print(f"Memory-efficient storage: ~{len(stats_data)*2:.1f} KB for histograms vs potential GB for raw particle data")
        
    except FileNotFoundError:
        logging.error(f'Statistics file {stats_file} not found. Run convert_gas first.')
    except Exception as e:
        logging.error(f'Error reading statistics: {e}')

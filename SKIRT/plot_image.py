import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from auto_stretch.stretch import Stretch
from scipy.integrate import simpson
import os
from tqdm import tqdm
import logging
from glob import glob
from My_Plugin.skirt.convolve import make_convolved_image, make_rgb_fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.basicConfig(level=logging.INFO)
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

def get_band_info(band):
    wav_pair = {
        'optical': [0.4, 0.7, 'inferno'],
        'MIR': [5, 40, 'hot'],
        'FIR': [1e2, 1e4, 'hot'],
        }
    return wav_pair[band]

def integrate_image(hdu, wavs, band, dynamic_range=1e4):
    logging.info(f'Integrating between {band[0]} to {band[1]} micron...')
    
    wav_min = band[0]
    wav_max = band[1]
    wav_index_min = np.argmin(np.abs(wavs - wav_min))
    wav_index_max = np.argmin(np.abs(wavs - wav_max))

    # optical
    wavelength = wavs[wav_index_min:wav_index_max] 
    wavelength = np.expand_dims(wavelength, axis=(1, 2))
    wavelength = np.repeat(wavelength, hdu.data.shape[1], axis=1)
    wavelength = np.repeat(wavelength, hdu.data.shape[2], axis=2)
    wavelength = wavelength[:,:,:,0]
    image = simpson(hdu.data[wav_index_min:wav_index_max,:,:], x=wavelength, axis=0)
    image[image<np.max(image)/dynamic_range] = np.max(image)/dynamic_range
    return image

def one_fig(hdul, band, fname, dynamic_range=1e4):
    #band_info = get_band_info(band)
    x_center = hdul[0].header['CRVAL1']
    x_scale  = hdul[0].header['CDELT1']
    x_npixel = hdul[0].header['CRPIX1']
    x_unit   = hdul[0].header['CUNIT1']
    y_center = hdul[0].header['CRVAL2']
    y_scale  = hdul[0].header['CDELT2']
    y_npixel = hdul[0].header['CRPIX2']
    y_unit   = hdul[0].header['CUNIT2']
    B_unit   = hdul[0].header['BUNIT']

    wavs = np.array(hdul[1].data.tolist())
    
    #image = integrate_image(hdu, wavs, band_info[:2], dynamic_range) 
    image = make_convolved_image(hdul, band, dynamic_range)
    fig, ax = plt.subplots(figsize=(6, 6*920/1085))
    #unit = 1e6*u.Jy/u.sr*u.Hz
    plot = ax.imshow(
        np.fliplr(image),
        #cmap=band_info[2], 
        cmap='gray', 
        norm=LogNorm(),
        #norm=Normalize(), 
        origin='upper',
        extent=[
            x_center - x_scale*x_npixel,
            x_center + x_scale*x_npixel,
            y_center - y_scale*y_npixel,
            y_center + y_scale*y_npixel,
            ]
        )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.)
    cbar = plt.colorbar(plot, cax=cax)
    #cbar.set_label(rf'Intensity ({B_unit})', rotation=90, labelpad=15)
    cbar.set_label(rf'Intensity (arbitary)', rotation=90, labelpad=15)

    ax.set_xlabel(rf'$x$ ({x_unit})')
    ax.set_ylabel(rf'$y$ ({y_unit})')
    #ax.text(0.05, 0.05, f'{band} ({band_info[0]} - {band_info[1]} micron)', ha='left', va='bottom', transform=ax.transAxes, color='w')
    band_annotation = ' '.join(band.split('_'))#.title()
    ax.text(0.05, 0.05, band_annotation, ha='left', va='bottom', transform=ax.transAxes, color='w')
    plt.tight_layout()
    logging.info(f'Saving {band} image on {fname}')
    fig.savefig(fname, dpi=300)
    plt.close()

def one_fits(path, dynamic_range=1e4):
    logging.info(f'Working on {path}')
    hdul = fits.open(path)
    bands = [
        #'ALMA_ALMA_9',
        'HERSCHEL_PACS_100', 
        #'SPITZER_IRAC_I4',
        #'RUBIN_LSST_Y',
        #'GENERIC_JOHNSON_R',
        'GENERIC_JOHNSON_V',
        #'GENERIC_JOHNSON_B',
        #'GALEX_GALEX_NUV'
        ]
    #for band in bands:
    #    fname = path.split('.')[0] + f'_{band}.png'
    #    one_fig(hdul, band, fname, dynamic_range)
    make_rgb_fits(hdul, ['GENERIC_JOHNSON_R', 'GENERIC_JOHNSON_V', 'GENERIC_JOHNSON_B'], path.split('.')[0]+'_optical_rgb.fits')
    make_rgb_fits(hdul, ['HERSCHEL_PACS_160', 'HERSCHEL_PACS_100', 'HERSCHEL_PACS_70'],  path.split('.')[0]+'_FIR_rgb.fits')
    # Join band names with commas for the command
    #bands_str = ",".join(bands)
    #command = f'python -m pts.do plot_bands --names="{bands_str}"'
    #os.system(command)
    return

def data_cube_movie(path, dynamic_range=1e3):
    hdul = fits.open(path)
    wavs = hdul[1].data
    
    save_path = './' + path.split('.')[0] + f'_data_cube/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for i, wav in tqdm(enumerate(wavs)):
        img = hdul[0].data[i,:,:]
        img[img<np.max(img)/dynamic_range] = np.max(img)/dynamic_range
        fig, ax = plt.subplots()
        p = ax.imshow(img, cmap='gray', norm=LogNorm())
        plt.colorbar(p)
        ax.axis('off')
        ax.text(0.05, 0.05, f'{wav[0]:.2f} micron', ha='left', va='bottom', transform=ax.transAxes, color='w')
        fig.savefig(save_path+f'{i:04d}.png', dpi=300)
        plt.close()
    return None

if __name__ == '__main__':
    paths = sorted(list(glob('*total.fits')))
    for path in paths:
        one_fits(path)
        #data_cube_movie(path)

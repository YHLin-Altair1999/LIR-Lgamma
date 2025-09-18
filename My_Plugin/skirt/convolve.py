import numpy as np
import os
import pts.storedtable as stab
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from auto_stretch.stretch import Stretch
import logging
logging.basicConfig(level=logging.INFO)

def integrate_datacube_filter(datacube, wave1, filter_wave, filter_response, dynamic_range=1e4):
    """
    Integrate a spectral datacube with a filter response function.
    
    Parameters:
    -----------
    datacube: np.ndarray, shape (n_y, n_x, n_wave1)
        3D array where first two dimensions are spatial and third is spectral
    wave1: np.ndarray, shape (n_wave1,)
        Wavelength axis for the datacube
    filter_wave: np.ndarray, shape (n_wave2,)
        Wavelength points for the filter response function
    filter_response: np.ndarray, shape (n_wave2,)
        Filter response values
    method: str
        Integration method: 'trapz' or 'simpson'
        
    Returns:
    --------
    filtered_image: np.ndarray, shape (n_y, n_x)
        2D image after integrating each spectrum with the filter
    """
    # Input validation
    if datacube.ndim != 3:
        raise ValueError("Datacube must be 3D array")
    if wave1.shape[0] != datacube.shape[0]:
        raise ValueError("Wavelength axis must match datacube spectral dimension")
        
    # Get spatial dimensions
    n_wave, n_y, n_x = datacube.shape
    
    # Create interpolated filter function
    # Use linear interpolation for filter as it's typically smooth
    filter_interp = interp1d(filter_wave.to('micron').value, filter_response.to('micron**(-1)').value,
                            kind='linear',
                            bounds_error=False,
                            fill_value=0.0)
    
    # Interpolate filter onto datacube wavelength grid
    filter_on_wave1 = filter_interp(wave1.to('micron').value).squeeze()
    
    # Broadcast filter to match datacube shape for efficient multiplication
    filter_3d = filter_on_wave1[:, np.newaxis, np.newaxis]
    
    # Multiply datacube by filter
    product = datacube * filter_3d
    
    filtered_image = simpson(product, x=wave1.squeeze().to('micron').value, axis=0)
    filtered_image[filtered_image<np.max(filtered_image)/dynamic_range] = np.max(filtered_image)/dynamic_range
    
    return filtered_image

def make_convolved_image(hdul, band, dynamic_range=1e4):
    wavs = np.array(hdul[1].data.tolist())*u.micron
    cube = hdul[0].data
    band_base_folder = '/tscc/nfs/home/yel051/codes/SKIRT/resources/SKIRT9_Resources_Core/Band/'
    band = stab.readStoredTable(band_base_folder + f'{band}_BroadBand.stab')
    image = integrate_datacube_filter(cube, wavs, band['lambda'], band['T'], dynamic_range)
    return image

def make_rgb_fits(hdul, bands, filename, normalize=True):
    header = hdul[0].header
    x_center = header['CRVAL1']
    x_scale  = header['CDELT1']
    x_npixel = header['CRPIX1']
    x_unit   = header['CUNIT1']
    y_center = header['CRVAL2']
    y_scale  = header['CDELT2']
    y_npixel = header['CRPIX2']
    y_unit   = header['CUNIT2']
    B_unit   = header['BUNIT']
    d = header['DISTANGD']*u.Unit(header['DISTUNIT'])
    dtheta = header['CDELT1']*u.Unit(header['CUNIT1'])
    box_size = d*header['NAXIS1']*dtheta.to('rad').value

    r = make_convolved_image(hdul, bands[0])
    g = make_convolved_image(hdul, bands[1])
    b = make_convolved_image(hdul, bands[2])
    rgb = np.stack([r, g, b], axis=0)
    if normalize:
        rgb = rgb / np.max(rgb)
    
    #fig, ax = plt.subplots(figsize=(6, 6*920/1085))
    fig, ax = plt.subplots(figsize=(6, 6))
    # Remove all padding and margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # Turn off axis lines, ticks, and labels
    ax.set_axis_off()
    
    image = np.stack([i**0.2 for i in [r, g, b]], axis=2)
    image /= np.max(image)

    #image = np.stack([i for i in [r, g, b]], axis=2)
    #image /= np.max(image)
    #image = Stretch().stretch(image)

    image = np.stack([i for i in [r, g, b]], axis=2)
    image = np.log10(image)
    image -= np.min(image)
    image /= np.max(image)
    
    plot = ax.imshow(
        np.fliplr(image),
        origin='upper',
        extent=[
            x_center - x_scale*x_npixel/2,
            x_center + x_scale*x_npixel/2,
            y_center - y_scale*y_npixel/2,
            y_center + y_scale*y_npixel/2,
            ]
        )
    
    # Define font size and calculate line height based on it
    font_size = 12  # You can adjust this value
    line_height = font_size * 1. / 300  # Convert points to figure fraction (assuming 72 DPI)
    base_x, base_y = 0.05, 0.05
    
    # Add annotations in a way that doesn't affect margins
    ax.text(base_x, base_y, f"{bands[0].replace('_', ' ')}", ha='left', va='bottom', 
            transform=ax.transAxes, color='#FF0000', fontsize=font_size)
    ax.text(base_x, base_y + line_height, f"{bands[1].replace('_', ' ')}", ha='left', va='bottom', 
            transform=ax.transAxes, color='#92D050', fontsize=font_size)
    ax.text(base_x, base_y + 2*line_height, f"{bands[2].replace('_', ' ')}", ha='left', va='bottom', 
            transform=ax.transAxes, color='#00B0F0', fontsize=font_size)

    ax.text(0.95, 0.05, f"Box size: {box_size.to('kpc'):.0f}", ha='right', va='bottom', 
            transform=ax.transAxes, color='w', fontsize=font_size)

    ax.text(0.05, 0.95, str(os.getcwd()).split('/')[-2].replace('_', ' '), transform=ax.transAxes, fontsize=14, color='w', ha='left', va='top')

    png_fname = filename.split('.')[0] + '.png'
    logging.info(f'Saving RGB image on {png_fname}')
    fig.savefig(png_fname, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Create HDU and save
    hdu = fits.PrimaryHDU(rgb)
    hdu.header['NAXIS3'] = 3  # Indicate this is a 3-channel image
    hdu.writeto(filename, overwrite=True)
    return

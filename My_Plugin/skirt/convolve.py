import numpy as np
import os
import pts.storedtable as stab
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as fits
from scipy.interpolate import interp1d
from scipy.integrate import simpson

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
    r = make_convolved_image(hdul, bands[0])
    g = make_convolved_image(hdul, bands[1])
    b = make_convolved_image(hdul, bands[2])
    rgb = np.stack([r, g, b], axis=0)
    if normalize:
        rgb = rgb / np.max(rgb)
    # Create HDU and save
    hdu = fits.PrimaryHDU(rgb)
    hdu.header['NAXIS3'] = 3  # Indicate this is a 3-channel image
    hdu.writeto(filename, overwrite=True)
    return

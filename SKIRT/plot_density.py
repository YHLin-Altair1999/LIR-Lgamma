import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
hdul = fits.open('highSNR_dns_dust_rho_xy.fits')

img = plt.imshow(hdul[0].data, cmap='inferno', norm=LogNorm())
plt.colorbar(img)
plt.savefig('density.png', dpi=300)



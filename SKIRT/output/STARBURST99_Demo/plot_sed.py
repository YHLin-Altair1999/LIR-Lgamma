import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.constants as c
import astropy.units as u
from glob import glob
import os

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

names = ['wavelength', 'total', 'transparent', 'direct', 'scattered', 'direct secondary', 'scattered secondary', 'transparent secondary']
def one_sed(fname, x_unit='micron'):
    df = pd.read_csv(fname, names=names, skiprows=9, sep=' ')
    w = np.array(df['wavelength'])*u.micron
    nu = c.c/w
    fig, ax = plt.subplots(figsize=(5,3))
    ax.loglog(
        w.to(x_unit).value, nu.to('Hz').value*df['total'],
        label='total', color='b', linewidth=5, linestyle='solid', zorder=3, alpha=0.5
        )
    '''
    ax.loglog(
        w.to(x_unit).value, nu.to('Hz').value*df['transparent'],
        label='primary transparent', linestyle='dashed', color='b'
        )
    ax.loglog(
        w.to(x_unit).value, nu.to('Hz').value*df['direct secondary'], 
        label='direct secondary', color='red', linestyle='dashed'
        )
    '''
    if fname.split('_')[0] == 'bc':
        name = 'BruzualCharlot'
    elif fname.split('_')[0] == 'sb99':
        name = 'Starburst99'
    else:
        name = 'Unknown'
    ax.text(0.05, 0.1, name, transform=ax.transAxes, fontsize=12, color='k', ha='left', va='bottom')
    ax.legend()
    ax.set_xlim(1e-1,1e3)
    #ax.set_ylim(bottom=1e14)
    #ax.set_ylim(1e12, 3e17)
    ax.set_xlabel(r'Wavelength (micron)')
    ax.set_ylabel(r'$\nu F_{\nu} ~{\rm (Hz \cdot Jy)}$')
    plt.tight_layout()
    fig.savefig('_'.join(fname.split('_')[:-1])+'_SED.png', dpi=300)

if __name__ == '__main__':
    fnames = list(glob('*_sed.dat'))
    for fname in fnames:
        one_sed(fname)

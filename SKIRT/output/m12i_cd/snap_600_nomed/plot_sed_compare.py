import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.constants as c
import astropy.units as u
from glob import glob

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

def one_sed(fname, ax, i, x_unit='micron'):
    ages = np.array([0.001,0.005,0.01,0.2,0.4])*1e3
    #age = float(fname.split('_')[-1][:-4])
    #print(age)
    names = ['wavelength', 'total']
    df = pd.read_csv(fname, names=names, skiprows=3, sep=' ')
    w = np.array(df['wavelength'])*u.micron
    nu = c.c/w
    cmap = plt.get_cmap('seismic')
    ax.loglog(
        w.to(x_unit).value, nu.to('Hz').value*df['total'],
        label=f'{ages[i]:.0f} Myr', color=cmap(i/5), linewidth=2, linestyle='solid', zorder=3, alpha=0.8
        )
    return ax

if __name__ == '__main__':
    fnames = list(glob('*_sed*.dat'))
    fig, ax = plt.subplots(figsize=(5,3))
    for i, fname in enumerate(fnames):
        one_sed(fname, ax, i)
    ax.legend()
    ax.set_xlim(1e-1,1e3)
    ax.set_ylim(bottom=1e14)
    ax.set_xlabel(r'Wavelength (micron)')
    ax.set_ylabel(r'$\nu F_{\nu} ~{\rm (Hz \cdot Jy)}$')
    plt.tight_layout()
    fig.savefig('nomed_SED_compare.png', dpi=300)

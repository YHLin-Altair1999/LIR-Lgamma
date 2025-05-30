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
def one_sed(fname, x_unit=u.micron, yunit=u.Unit('erg*s**-1*cm**-2')):
    df = pd.read_csv(fname, names=names, skiprows=9, sep=' ')
    w = np.array(df['wavelength'])*u.micron
    total_F_lambda       = ((np.array(df['total'])*u.Jy) * c.c / w**2) * w
    transparent_F_lambda = ((np.array(df['transparent'])*u.Jy) * c.c / w**2) * w
    secondary_F_lambda   = ((np.array(df['direct secondary'])*u.Jy) * c.c / w**2) * w
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(
        w.to(x_unit).value, total_F_lambda.to(yunit).value,
        label='total', color='b', linewidth=5, linestyle='solid', zorder=3, alpha=0.5
        )
    ax.plot(
        w.to(x_unit).value, transparent_F_lambda.to(yunit).value,
        label='primary transparent', linestyle='dashed', color='b'
        )
    ax.plot(
        w.to(x_unit).value, secondary_F_lambda.to(yunit).value,
        label='direct secondary', color='red', linestyle='dashed'
        )
    ax.text(0.05, 0.9, str(os.getcwd()).split('/')[-2].replace('_', ' '), transform=ax.transAxes, fontsize=12, color='k', ha='left', va='top')
    ax.legend()
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlim(1e-1,1e3)
    ax.set_ylim(bottom=1e-14)
    #ax.set_ylim(1e12, 3e17)
    ax.set_xlabel(rf'Wavelength ({x_unit.to_string(format="latex_inline")})')
    ax.set_ylabel(rf'$\lambda F_{{\lambda}}$ ({yunit.to_string(format="latex_inline")})')
    plt.tight_layout()
    fig.savefig(''.join(fname.split('_')[:-1])+'_SED.png', dpi=300)

if __name__ == '__main__':
    fnames = list(glob('*_sed.dat'))
    for fname in fnames:
        one_sed(fname)

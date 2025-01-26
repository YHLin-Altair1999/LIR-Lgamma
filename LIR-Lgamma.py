import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from My_Plugin.quantity import L_IR, L_gamma
import yt
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
import astropy.constants as c
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

def plot_sim(snap, ax):
    sed_path = f'/tscc/lustre/ddn/scratch/yel051/SKIRT/output/snap_{snap}/run_SKIRT_i00_sed.dat'
    fname = f'/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output/snapdir_{snap:03d}/'
    ds = yt.load(fname)
    LIR = L_IR(sed_path)
    Lgamma = L_gamma(ds)
    ax.scatter(LIR.to('L_sun'), Lgamma, label=f'm12i snap {snap}', marker='*')
    return ax

def plot_obs(ax):
    e_ph = 1*u.GeV
    obs_path = './obs_data/Ambrosone_2024.csv'
    df = pd.read_csv(obs_path)
    df['L_IR (L_sun)'] = np.array(df['LIR'])*1e10
    df['z'] = z_at_value(Planck18.luminosity_distance, np.array(df['DL'])*u.Mpc)
    df['L_gamma (erg/s)'] = (
        (4*np.pi*(np.array(df['DL'])*u.Mpc)**2 / (1+np.array(df['z']))**(2-np.array(df['gamma'])) * 
        np.array(df['F1-1000 GeV'])*1e-10*e_ph/(u.cm**2*u.s)).to('erg/s').value
        )

    ax.scatter(df['L_IR (L_sun)'], df['L_gamma (erg/s)'], label='Ambrosone et al. (2024)', color='C0', marker='^')
    return ax

def finalize(fig, ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$L_{\rm IR} ~(L_\odot)$')
    ax.set_ylabel(r'$L_\gamma ~{\rm (erg/s)}$')
    ax.legend()
    plt.tight_layout()
    fig.savefig('LIR-Lgamma.png', dpi=300)

def main():
    #snaps = np.arange(100,600,50)
    #snaps[0] += 2
    snaps = [100, 600]
    fig, ax = plt.subplots(figsize=(5,3))
    for snap in snaps:
        plot_sim(snap, ax)
    plot_obs(ax)
    finalize(fig, ax)

if __name__ == '__main__':
    main()


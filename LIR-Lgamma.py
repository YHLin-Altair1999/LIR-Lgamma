import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from My_Plugin.quantity import L_IR, L_gamma_yt, L_gamma_YHLin
from My_Plugin.LoadData import get_snap_path, get_center
import yt
import os
from glob import glob
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
import astropy.constants as c
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

def calculate_Lgamma(galaxy, snap, mode='yt', aperture=25*u.kpc):
    if mode == 'yt':
        print('Calculating gamma ray luminosity using yt')
        path = get_snap_path(galaxy, snap)
        ds = yt.load(glob(os.path.join(path, "*.hdf5"))[0])
        out = L_gamma_yt(ds, get_center(galaxy, snap))
    else:
        out = L_gamma_YHLin(galaxy, snap, aperture)
    return out

def calculate_LIR(galaxy, snap):
    sed_path = f'/tscc/lustre/ddn/scratch/yel051/SKIRT/output/{galaxy}/snap_{snap}/run_SKIRT_i00_sed.dat'
    LIR = L_IR(sed_path)
    print(LIR)
    return LIR

def make_table(galaxy, snaps, table_path):
    data = []
    for snap in snaps:
        data.append({
            'galaxy': galaxy, 
            'snap': snap, 
            'L_gamma (erg/s)': calculate_Lgamma(galaxy, snap, mode='YHLin', aperture=5*u.kpc).to('erg/s').value, 
            'L_IR (L_sun)': calculate_LIR(galaxy, snap).to('L_sun').value
            })

    if os.path.exists(table_path):
        existing_df = pd.read_csv(table_path)
        new_df = pd.DataFrame(data)
        df = pd.concat([existing_df, new_df]).drop_duplicates(subset='galaxy', keep='last')
    else:
        df = pd.DataFrame(data)
    df = df.sort_values(by='snap')
    df.to_csv(table_path, index=False)

def plot_sim(table_path, ax):
    df = pd.read_csv(table_path)
    for i in range(df.shape[0]):
        ax.scatter(df['L_IR (L_sun)'][i], df['L_gamma (erg/s)'][i], label=df['galaxy'][i], marker='*', color=f'C{i}', zorder=3, s=60)
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

    ax.scatter(
        df['L_IR (L_sun)'], df['L_gamma (erg/s)'], 
        label='Ambrosone et al. (2024)', color='C0', marker='^', edgecolor='None', alpha=0.5)
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
    table_path = './tables/Lgamma_LIR.csv'
    '''
    #galaxies = ['m12i_et', 'm12i_sc_fx10', 'm12i_sc_fx100']
    galaxies = ['m12i_cd']
    snaps = [600]
    for galaxy in galaxies:
        make_table(galaxy, snaps, table_path)
    '''
    fig, ax = plt.subplots(figsize=(5,4))
    plot_sim(table_path, ax)
    plot_obs(ax)
    finalize(fig, ax)

if __name__ == '__main__':
    main()


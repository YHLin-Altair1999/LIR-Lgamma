import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from My_Plugin.quantity import L_IR, L_gamma_yt
import yt
import os
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
import astropy.constants as c
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

def calculate_Lgamma(snap, mode='yt'):
    fname = f'/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output/snapdir_{snap:03d}/'
    if mode == 'yt':
        ds = yt.load(fname)
        out = L_gamma_yt(ds)
    else:
        out = Lgamma_LYH(snap)
    return out

def Lgamma_YHL(snap):
    print('this part is not done yet!')
    L_gamma = 0
    return L_gamma

def calculate_LIR(snap):
    sed_path = f'/tscc/lustre/ddn/scratch/yel051/SKIRT/output/snap_{snap}/run_SKIRT_i00_sed.dat'
    LIR = L_IR(sed_path)
    return LIR

def make_table(snaps, table_path):
    data = []
    for snap in snaps:
        data.append({'snap': snap, 'L_gamma': calculate_Lgamma(snap), 'L_IR': calculate_LIR(snap)})

    if os.path.exists(table_path):
        existing_df = pd.read_csv(table_path)
        new_df = pd.DataFrame(data)
        df = pd.concat([existing_df, new_df]).drop_duplicates(subset='snap', keep='last')
    else:
        df = pd.DataFrame(data)
    df = df.sort_values(by='snap')
    df.to_csv(table_path, index=False)

def plot_sim(table_path, ax):
    df = pd.read_csv(table_path)
    LIR = np.array(df['L_IR'])*((u.erg/u.s)/u.L_sun).to('')
    Lgamma = np.array(df['L_gamma'])
    ax.scatter(LIR, Lgamma, label=f'm12i', marker='*', color='C1')
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
    table_path = './tables/Lgamma_LIR.csv'
    #snaps = np.arange(100,600,50)
    #snaps[0] += 2
    #snaps = [100, 600]
    snaps = [600]
    #make_table(snaps, table_path)
    fig, ax = plt.subplots(figsize=(5,4))
    plot_sim(table_path, ax)
    plot_obs(ax)
    finalize(fig, ax)

if __name__ == '__main__':
    main()


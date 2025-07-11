import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import My_Plugin.quantity as q
#from My_Plugin.quantity import L_IR, L_gamma_yt, L_gamma_YHLin, L_gamma_make_one_profile, L_gamma_make_one_profile_Pfrommer
from My_Plugin.LoadData import get_snap_path, get_center
import yt
import os
from glob import glob
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
import astropy.constants as c
from tqdm import tqdm
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

def make_profiles(inputs, rs, sfr_type='FIR'):
    for galaxy in inputs.keys():
        for snap in inputs[galaxy]:
            q.SFR_make_one_profile(galaxy, snap, rs, sfr_type=sfr_type)

def plot_one_profile(ax, galaxy, snap, xunit, yunit, plot_type='cumulative'):
    target_folder = '/tscc/lustre/ddn/scratch/yel051/tables/SFR_profiles'
    fname = os.path.join(target_folder, f'SFR_profile_{galaxy}_snap{snap:03d}.npy')
    profile = np.load(fname)
    r = profile[:,0]*u.cm
    dr = r[1]-r[0]
    SFR = profile[:,1]*u.M_sun/u.yr
    SFR_cumulative = np.cumsum(SFR)

    if galaxy.startswith('m12i_'):
        linestyle = 'solid'
    else:
        linestyle = 'dashed'
    if plot_type == 'cumulative':
        ax.semilogy(
            r.to(xunit).value, SFR_cumulative.to(yunit).value,
            linestyle=linestyle, label=f'{galaxy}')
    elif plot_type == 'differential':
        ax.semilogy(
            r.to(xunit).value, SFR.to(yunit).value/dr.to(xunit).value, 
            linestyle=linestyle, label=f'{galaxy}')
    else:
        raise ValueError("plot_type must be 'cumulative' or 'differential'")
    return ax

def plot_profiles(inputs, plot_type='cumulative'):
    fig, ax = plt.subplots(figsize=(6,4))
    xunit = u.kpc
    yunit = u.M_sun/u.yr
    for galaxy in tqdm(inputs.keys()):
        for snap in inputs[galaxy]:
            plot_one_profile(ax, galaxy, snap, xunit=xunit, yunit=yunit, plot_type=plot_type)
    xmin = 0.0*u.kpc / xunit
    ax.set_xlim(left=xmin)
    ax.set_ylim(1e-5, 3e0)
    ax.set_xlabel(rf'Radius ~({xunit.to_string('latex_inline')})')
    ax.set_ylabel(rf'SFR ~({yunit.to_string('latex_inline')})')
    ax.set_title(rf'SFR profile')
    #ax.legend()
    plt.tight_layout()
    fig.savefig('SFR_profile.png', dpi=300)
    return

if __name__ == '__main__':
    inputs = {
        'm12i_et': [60], 
        'm12i_sc_fx10': [60], 
        'm12i_sc_fx100': [60],
        'm12i_cd': [600, 590, 585, 580],
        'm11b_cd': [600],
        'm11c_cd': [600],
        'm11d_cd': [600],
        'm11f_cd': [600],
        'm11g_cd': [600],
        'm11h_cd': [600],
        'm11v_cd': [600],
        'm10v_cd': [600],
        'm09_cd': [600],
        'm11f_et_AlfvenMax': [600],
        'm11f_et_FastMax': [600],
        'm11f_sc_fcas50': [600]
        }
    rs = np.linspace(0, 25, 50)*u.kpc
    #rs = np.logspace(-5, 1, 20)*u.kpc
    make_profiles(inputs, rs, sfr_type='Ha')
    #plot_profiles(inputs, plot_type='cumulative')


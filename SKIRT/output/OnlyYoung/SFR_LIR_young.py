import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import os
from glob import glob
from My_Plugin.quantity import L_IR

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    })

SFR_folder = '/tscc/lustre/ddn/scratch/yel051/tables/SFR'

fnames = glob('./**/run_SKIRT_i00_sed.dat', recursive=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

# Store data for ratio calculation
data_points = []

for fname in fnames:
    LIR = L_IR(fname)
    galaxy = fname.split('/')[1]
    snap = int(fname.split('/')[2].split('_')[1])
    SFR_fname = os.path.join(SFR_folder, f'SFR_{galaxy}_snap{snap:03d}.npy')
    SFR, SFR_err = np.load(SFR_fname)*u.Msun/u.yr
    
    # Plot on first axis
    ax1.errorbar(SFR.to('Msun/yr').value, LIR.to('Lsun').value, 
                xerr=SFR_err.to('Msun/yr').value, fmt='o', label=galaxy, color='C0', alpha=0.7)
    
    # Store data for ratio calculation
    data_points.append({
        'galaxy': galaxy,
        'SFR': SFR,
        'SFR_err': SFR_err,
        'LIR': LIR
    })

def SFR2LIR(SFR, epsilon=0.79):
    """Convert SFR to LIR using the Kennicutt relation"""
    return (SFR/(u.Msun/u.yr))/(epsilon*1.7e-10)*u.L_sun # value used in Pfrommer17, Chabrier IMF
    #return (SFR/(u.Msun/u.yr))*5.8e9*u.L_sun # original value used in K98, Salpeter IMF

SFR_line = np.linspace(1e-4, 1e1, 1000)* u.Msun/u.yr
ax1.plot(SFR_line.to('Msun/yr').value, SFR2LIR(SFR_line).to('L_sun').value,
         linestyle='--', color='black', label='Kennicutt relation')

ax1.set_title(r'Young stars ($< 200$ Myr) only')
ax1.set_yscale('log')
ax1.set_ylabel(r'$L_{\rm IR}$ [L$_{\odot}$]')
ax1.set_xlim(1e-1,1e1)
ax1.set_ylim(1e8,1e11)
#ax1.legend()

# Second subplot: ratio plot
for data in data_points:
    LIR_predicted = SFR2LIR(data['SFR'])
    ratio = data['LIR'] / LIR_predicted
    # Error propagation for ratio
    ratio_err = ratio * data['SFR_err'] / data['SFR']
    
    ax2.errorbar(data['SFR'].to('Msun/yr').value, ratio.to('').value, 
                yerr=ratio_err.to('').value, fmt='o', label=data['galaxy'], color='C0', alpha=0.7)

# Add horizontal line at ratio = 1
ax2.axhline(y=1, linestyle='--', color='black', alpha=0.7)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(1e-1,1e1)
ax2.set_ylim(8e-2,3.5e0)
ax2.set_xlabel(r'SFR [M$_{\odot}$ yr$^{-1}$]')
ax2.set_ylabel(r'$L_{\rm IR}$ / $L_{\rm IR, K98}$')
plt.tight_layout()
fig.savefig(f'SFR_LIR_young.png', dpi=300)
plt.close()

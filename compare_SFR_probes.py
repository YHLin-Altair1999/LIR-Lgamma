import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.constants as c

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

fnames = [
    './tables/SFR_LIR_100Myr.csv',
    './tables/SFR_LIR_2Myr.csv',
    './tables/SFR_LIR_Gas.csv'
    ]

fig, ax = plt.subplots(figsize=(5,3))
for i, fname in enumerate(fnames):
    df = pd.read_csv(fname)
    # Filter out rows with NaN values in 'L_IR (L_sun)' or 'SFR (M_sun/yr)'
    df = df.dropna(subset=['L_IR (L_sun)', 'SFR (M_sun/yr)'])
    label = fname.split('/')[-1].split('.')[0].split('_')[-1]
    ax.scatter(
        df['SFR (M_sun/yr)'], df['L_IR (L_sun)'], 
        label=label, color=f'C{i}', alpha=0.5, s=50, edgecolor='None'
        )
    if i == 0:
        for j in range(len(df)):
            ax.annotate(
                    df['galaxy'].iloc[j], 
                    (df['SFR (M_sun/yr)'].iloc[j], df['L_IR (L_sun)'].iloc[j]),
                    xytext=(5, 0),  # Small offset from the point
                    textcoords='offset points',
                    fontsize=8,
                    color=f'C{i}'
                    )

SFR_range = np.logspace(-5, 1, 100)*u.Msun/u.yr
epsilon = 0.79
LIR_Kennicutt = (SFR_range/(u.M_sun/u.yr))/(epsilon*1.7e-10)*u.L_sun

ax.plot(
    SFR_range.to('M_sun/yr').value, 
    LIR_Kennicutt.to('L_sun').value, 
    color='k', linestyle='--', label=r'Kennicutt (1998)'
    )

# Set axis labels and scales
ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_ylim(1e7, 1e11)
ax.set_xlabel(r'$\mathrm{SFR}~(M_\odot/\mathrm{yr})$')
ax.set_ylabel(r'$L_{\rm IR, ~8-1000 ~\mu m} ~(L_\odot)$')
ax.legend()
plt.tight_layout()
fig.savefig('./SFR_LIR_compareSFRprobes.png', dpi=300)
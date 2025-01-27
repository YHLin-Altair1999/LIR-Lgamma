import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

names = ['x', 'y', 'z', 'l', 'M', 'Z', 'age']
D = pd.read_csv('./stars.txt', skiprows=12, sep=' ', names=names)

n_bins = 500
t_max = 13.8 # Gyr
delta_t = t_max*1e9/n_bins # yr/bin
hist, bin_edges = np.histogram(D['age'], bins=n_bins+1, range=(0-delta_t/1e9,t_max), weights=D['M']/delta_t)

fig, ax = plt.subplots(figsize=(4,3))
ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
ax.set_xlabel(r'Lookback time (Gyr)')
ax.set_ylabel(r'Star Formation Rate (SFR, $M_\odot/{\rm yr}$)')
plt.tight_layout()
fig.savefig('SFH.png', dpi=300)


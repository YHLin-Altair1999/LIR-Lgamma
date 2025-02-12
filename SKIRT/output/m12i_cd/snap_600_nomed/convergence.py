import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from My_Plugin.quantity import L_IR
import astropy.units as u
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

fnames = glob('./run_SKIRT_level*_sed.dat')
LIRs = np.array([L_IR(fname).to('erg/s').value for fname in fnames] + [1.9136947307956677e44])
levels = np.array([int(fname.split('level')[-1].split('_')[0]) for fname in fnames] + [10])
fig, ax = plt.subplots()
ax.scatter(levels, LIRs)
ax.set_xlabel('Maximum refinement level')
ax.set_ylabel(r'$L_\gamma ~{\rm (erg/s)}$')
ax.set_title(r'Y axis is $\pm$ 1 per cent median')
dy = 0.01
ax.set_ylim(0.99*np.median(LIRs), 1.01*np.median(LIRs))
fig.savefig('convergence.png', dpi=300)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
import astropy.constants as c
from glob import glob

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    }) 

names = ['wavelength', 'total', 'transparent', 'direct', 'scattered', 'direct secondary', 'scattered secondary', 'transparent secondary']

def one_curve(i, fig, ax, fname, norm_wav=0.55*u.micron):
    wav_max = 1*u.micron
    df = pd.read_csv(fname, names=names, skiprows=9, sep=' ')
    wav = np.array(df['wavelength'])*u.micron
    wav_max_index = np.argmin(np.abs(wav-wav_max))
    norm_index = np.argmin(np.abs(wav - norm_wav))
    df = df.iloc[:wav_max_index,:]
    A = 1.086*np.log(np.array(df['transparent']) / np.array(df['total']))
    A /= A[norm_index]
    if i == 0:
        color = 'k'
    else:
        color = f'C{i-1}'
    ax.plot(1/df['wavelength'], A, label=f'{fname.split('_')[1]}', color=color, linestyle='solid', alpha=0.8)
    return ax

if __name__ == '__main__':
    fnames = sorted(list(glob('*_sed.dat')))
    fig, ax = plt.subplots(figsize=(5,3))
    for i, fname in enumerate(fnames):
        one_curve(i, fig, ax, fname)
    ax.legend()
    ax.set_ylim(0,10)
    ax.set_xlim(1,10)
    ax.set_xlabel(r'$1/\lambda~(\mu m^{-1})$')
    ax.set_ylabel(r'$A_\lambda / A_V$')
    plt.tight_layout()
    fig.savefig('attenuation_curve.png', dpi=300)

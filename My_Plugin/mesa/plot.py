import numpy as np
import mesaPlot as mp
import matplotlib.pyplot as plt
import re

def add_burn(ax, p, name):
    data = getattr(p, name)
    rho = 10**data['logRho'] 
    T = 10**data['logT'] 
    ax.loglog(rho, T, color='gray', alpha=0.5)
    texts = ''.join(list(name)[1:-4]).capitalize() + ' Burn'
    #texts = re.sub(r"_(\w)(\w+)", lambda m: f"{m.group(1).upper()} {m.group(2)}", name)
    ax.text(rho[-1]/3, T[-1], texts, ha='right', va='center', color='gray')
    return ax

def add_others(ax, p):
    ax.loglog(10**p._psi4['logRho'],  10**p._psi4['logT'],   color='gray', alpha=0.5, linestyle='dashed')
    ax.text(1e1, 3e4, 'Degenerate', rotation=60, rotation_mode='anchor', color='gray')
    #ax.loglog(10**p._elect['logRho'], 10**p._elect['logT'],  color='gray', alpha=0.5, linestyle='dashed')
    #ax.loglog(10**p._gamma4['logRho'],10**p._gamma4['logT'], color='gray', alpha=0.5, linestyle='dashed')
    #ax.loglog(10**p._kap['logRho'],   10**p._kap['logT'],    color='gray', alpha=0.5, linestyle='dashed')
    #ax.loglog(10**p._opal['logRho'],  10**p._opal['logT'])
    #ax.loglog(10**p._scvh['logRho'],  10**p._scvh['logT'])
    #ax.text()
    return ax

def rhoT_plot(m, time_ind, fig, ax):
    m.loadProfile(num=time_ind)
    rho = 10**m.prof.logRho
    T = 10**m.prof.logT
    r = m.prof.logR
    ax.scatter(rho, T, c=r, cmap='Spectral_r', zorder=3)
    p = mp.plot(rcparams_fixed=False)
    p._loadBurnData()
    add_burn(ax, p, '_hburn')
    add_burn(ax, p, '_heburn')
    add_burn(ax, p, '_cburn')
    add_burn(ax, p, '_oburn')
    add_others(ax, p)
    ax.set_xlabel(r'Density $({\rm g~cm^{-3}})$')
    ax.set_ylabel(r'Temperature $({\rm K})$')
    ax.set_xlim(1e-10,1e11)
    ax.set_ylim(1e3,1e11)
    #plt.tight_layout()
    return fig, ax

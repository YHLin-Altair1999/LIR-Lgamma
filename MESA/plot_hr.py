import numpy as np
import mesaPlot as mp
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c

plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}", 'font.family': 'STIXGeneral'})
m=mp.MESA()

def one_plot(m, t):

    age = m.hist.star_age*u.yr
    index = np.argmin(np.abs((age - t).to('yr').value))

    fig, ax = plt.subplots(figsize=(5,4))

    T = 10**m.hist.log_Teff
    L = 10**m.hist.log_L

    ax.loglog(T, L)
    ax.loglog(T[index], L[index], marker='o', color='C1')

    ax.xaxis.set_inverted(True)
    ax.set_xlabel(r'Temperature (K)')
    ax.set_ylabel(r'Luminosity ($L_\odot$)')
    plt.tight_layout()
    plt.savefig('HR.png', dpi=300)

if __name__ == '__main__':
    #EXAMPLE_DIR = "./12M_pre_ms_to_core_collapse/LOGS/"
    EXAMPLE_DIR = "./1M_pre_ms_to_wd/LOGS_combined/"
    m.log_fold=EXAMPLE_DIR
    m.loadHistory(filename_in=EXAMPLE_DIR + 'history.data')
    one_plot(m, 0*u.Gyr)

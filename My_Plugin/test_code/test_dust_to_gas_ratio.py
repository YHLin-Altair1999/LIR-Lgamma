import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import os
from My_Plugin.skirt.convert import Dust_to_gas_ratio_RemyRuyer

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
})

nO_nH_ratio = np.logspace(-5, -2, 100)
D_G_ratio = 1/Dust_to_gas_ratio_RemyRuyer(nO_nH_ratio)

plt.semilogy(12 + np.log10(nO_nH_ratio), D_G_ratio)
plt.savefig('Gas_to_dust_ratio_RemyRuyer.png', dpi=300)

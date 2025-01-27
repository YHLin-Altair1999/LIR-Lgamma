import numpy as np
import mesaPlot as mp
import matplotlib.pyplot as plt
import re
from My_Plugin.mesa.plot import rhoT_plot

plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}", 'font.family': 'STIXGeneral'})
m=mp.MESA()
EXAMPLE_DIR = "./12M_pre_ms_to_core_collapse/LOGS"
m.log_fold=EXAMPLE_DIR

fig, ax = plt.subplots(figsize=(5,4))

rhoT_plot(m, 1000, fig=fig, ax=ax)
plt.savefig('TRho.png', dpi=300)

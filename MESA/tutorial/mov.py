import matplotlib.pyplot as plt
import mesaPlot as mp
import tulips
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import mesa_reader as mr

plt.rcParams.update({"text.usetex": True, 
                     "text.latex.preamble": r"\usepackage{amsmath}",
                     'font.family': 'STIXGeneral'})

EXAMPLE_DIR = "./LOGS/"

m = mp.MESA() # Create MESA object
m.loadHistory(filename_in=EXAMPLE_DIR + 'history.data')
m.log_fold=EXAMPLE_DIR
star_age = m.hist.star_age # Access star age
log_R = m.hist.log_R # Access star log radius

def one_fig(index):
    fig, axes = plt.subplots(figsize=(10,4.5), nrows=1, ncols=2)
    fig, axes[0] = tulips.perceived_color(m, time_ind=index, fig=fig, ax=axes[0])
    fig, axes[1] = tulips.energy_and_mixing(
        m, time_ind=0, cmin=-10, cmax=10, show_total_mass=True,
        show_mix=True, show_mix_legend=True, fig=fig, ax=axes[1])
    fig.savefig(f'./images/M1_{i:04d}.png', dpi=300)
    plt.close()

for i, index in tqdm(enumerate(range(0, len(star_age)))):
    one_fig(index)

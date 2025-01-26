import h5py
import matplotlib.pyplot as plt
import numpy as np

fname = '/tscc/lustre/ddn/scratch/yul232/m12i_cr_700/output/snapdir_600/snapshot_600.0.hdf5'
f = h5py.File(fname, 'r')

fig, ax = plt.subplots()
ax.scatter( f['PartType0']['NeutralHydrogenAbundance'], 
            f['PartType0']['temp']
            )



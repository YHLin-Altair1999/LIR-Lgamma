import numpy as np
import pandas as pd

header = open('/tscc/lustre/ddn/scratch/yel051/My_Plugin/skirt/skirt_header_stars.txt', 'r').read()
D = pd.read_csv('./stars_template.txt', sep=' ', skiprows=13, names=['x', 'y', 'z', 'l', 'M', 'Z', 't'])
D['SFR'] = np.where(D['t'] < 0.1e9, D['M'] / (D['t']), 0)
D['log_compactness'] = 5
D['covering_factor'] = 0.2
D['pressure'] = 1e11
D = D[['x', 'y', 'z', 'l', 'SFR', 'Z', 'log_compactness', 'pressure', 'covering_factor']]
np.savetxt("stars.txt", D, delimiter=" ", header=header)
#print(D)

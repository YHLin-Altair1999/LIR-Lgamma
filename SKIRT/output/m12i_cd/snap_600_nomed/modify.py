import pandas as pd
import numpy as np
from sys import argv

df = pd.read_csv('stars_template.txt', sep=' ', skiprows=12, names=['x', 'y', 'z', 'l', 'M', 'Z', 'age'])
output = df.to_numpy()
output = output[output[:,6]>float(argv[1])]
header = open('/tscc/lustre/ddn/scratch/yel051/My_Plugin/skirt/skirt_header_stars.txt', 'r').read()
np.savetxt("stars.txt", output, delimiter=" ", header=header)
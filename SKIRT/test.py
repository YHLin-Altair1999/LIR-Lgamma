import pts
import matplotlib.pyplot as plt
import pts.storedtable as stab
from scipy.interpolate import interp1d
import numpy as np

band = stab.readStoredTable('/tscc/nfs/home/yel051/codes/SKIRT/resources/SKIRT9_Resources_Core/Band/SLOAN_SDSS_U_BroadBand.stab')

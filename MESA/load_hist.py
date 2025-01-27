import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import mesaPlot as mp

def oneM_hist():
    DIRS = [
        "./1M_pre_ms_to_wd/LOGS_to_end_core_h_burn/",
        "./1M_pre_ms_to_wd/LOGS_to_start_he_core_flash/",
        "./1M_pre_ms_to_wd/LOGS_to_end_core_he_burn/",
        "./1M_pre_ms_to_wd/LOGS_to_end_agb/",
        "./1M_pre_ms_to_wd/LOGS_to_wd/"
        ]
    for i, DIR in enumerate(DIRS):
        print(DIR)
        if i == 0:
            m = mp.MESA()
            m.loadHistory(filename_in=DIR + 'history.data')
            print(m.hist.data.shape)
        else:
            m2 = mp.MESA()
            m2.loadHistory(filename_in=DIR + 'history.data')
            m.hist.data = np.concatenate((m.hist.data, m2.hist.data))
            print(m.hist.data.shape)
    return m



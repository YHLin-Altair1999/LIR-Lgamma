import numpy as np
import My_Plugin.quantity as q

if __name__ == '__main__':
    inputs = {
        #'m12f_cd': [600],
        #'m12i_et': [60], 
        #'m12i_sc_fx10': [60], 
        #'m12i_sc_fx100': [60],
        #'m12i_cd': np.arange(100, 600, 50),
        #'m12i_FIRE3_CRSpec_noBH': [60],
        'm12i_FIRE3_CRSpec': [60],
        #'m12i_FIRE3_CRSpec1': [60],
        #'m12r_cd': [600],
        #'m12w_cd': [600],
        #'m11b_cd': [600],
        #'m11b_cd_007': [600],
        #'m11b_cd_070': [600],
        #'m11b_cd_210': [600],
        #'m11c_cd': [600],
        #'m11d_cd': [600],
        #'m11f_cd': [600],
        #'m11g_cd': [600],
        #'m11h_cd': [600],
        #'m11v_cd': [600],
        #'m10v_cd': [600],
        #'m09_cd': [600],
        #'m11f_et_AlfvenMax': [600],
        #'m11f_et_FastMax': [600],
        #'m11f_sc_fcas50': [600]
        }
    for galaxy in inputs.keys():
        for snap in inputs[galaxy]:
            q.SFR_make_onezone(galaxy, snap)

from My_Plugin.SFR_LIR import SFR_LIR_Plot

galaxies = {
    'm12i_et': [60], 
    'm12i_sc_fx10': [60], 
    'm12i_sc_fx100': [60],
    'm12i_cd': [600],#, 590, 585, 580],
    'm12f_cd': [600],
    'm12r_cd': [600],
    'm12w_cd': [600],
    'm11b_cd': [600],
    #'m11b_cd_007': [600],
    #'m11b_cd_070': [600],
    #'m11b_cd_210': [600],
    'm11c_cd': [600],
    'm11d_cd': [600],
    'm11f_cd': [600],
    'm11g_cd': [600],
    'm11h_cd': [600],
    'm11v_cd': [600],
    #'m10v_cd': [600],
    'm11f_et_AlfvenMax': [600],
    'm11f_et_FastMax': [600],
    'm11f_sc_fcas50': [600]
}

plotter = SFR_LIR_Plot(
    galaxies=galaxies,
    output_filename='SFR_LIR_Full.png',)
plotter.run()


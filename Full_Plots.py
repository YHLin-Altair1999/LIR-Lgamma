import astropy.units as u
import astropy.constants as c
from My_Plugin.LIR_Lgamma import LIR_Lgamma_Plot
from My_Plugin.SFR_Lgamma import SFR_Lgamma_Plot

galaxies = {
        'm12f_cd': [600], 
        'm12i_et': [60], 
        'm12i_sc_fx10': [60], 
        'm12i_sc_fx100': [60],
        'm12i_cd': [600],
        #'m12i_cd': np.arange(100,600,50),
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
        #'m09_cd': [600],
        'm11f_et_AlfvenMax': [600],
        'm11f_et_FastMax': [600],
        'm11f_sc_fcas50': [600],
    }
E_min = 1*u.GeV
E_max = 1000*u.GeV

# Create the plotter with our parameters
plotter = LIR_Lgamma_Plot(
    galaxies=galaxies,
    E_min=E_min,
    E_max=E_max,
    show_sim_gal_name=False,
    show_obs_gal_name=False
    )

# Either run the full pipeline
plotter.run()

plotter = SFR_Lgamma_Plot(
    galaxies=galaxies,
    E_min=E_min, 
    E_max=E_max,
    show_obs_gal_name=False
    )
    
plotter.run()

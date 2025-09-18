import astropy.units as u
import astropy.constants as c
from My_Plugin.LIR_Lgamma import LIR_Lgamma_Plot
from My_Plugin.SFR_Lgamma import SFR_Lgamma_Plot

galaxies = {
        'm11b_cd_007': [600],
        'm11b_cd_070': [600],
        'm11b_cd_210': [600],
    }
E_min = 1*u.GeV
E_max = 1000*u.GeV

# Create the plotter with our parameters
plotter = LIR_Lgamma_Plot(
    galaxies=galaxies,
    E_min=E_min,
    E_max=E_max,
    show_sim_gal_name=False,
    show_obs_gal_name=False,
    show_calorimetric_limit=False,
    sed_base_path='/tscc/lustre/ddn/scratch/yel051/SKIRT/output/',
    output_filename='LIR_Lgamma_KappaCompare.png',
    )

# Either run the full pipeline
plotter.run()

plotter = SFR_Lgamma_Plot(
    galaxies=galaxies,
    E_min=E_min, 
    E_max=E_max,
    show_obs_gal_name=False,
    output_filename='SFR_Lgamma_KappaCompare.png'
    )
    
plotter.run()

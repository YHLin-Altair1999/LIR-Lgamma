import astropy.units as u
import astropy.constants as c
from My_Plugin.LIR_Lgamma import LIR_Lgamma_Plot
from My_Plugin.SFR_Lgamma import SFR_Lgamma_Plot

galaxies = {'m10v_cd': [600],}
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
    output_filename='LIR_Lgamma_ObsOnly.png',
    figsize=(4, 4)
    )

# Either run the full pipeline
plotter.run()

# Create the plotter with our parameters
plotter = LIR_Lgamma_Plot(
    galaxies=galaxies,
    E_min=E_min,
    E_max=E_max,
    show_sim_gal_name=False,
    show_obs_gal_name=True,
    show_calorimetric_limit=False,
    output_filename='LIR_Lgamma_ObsOnly_names.png',
    figsize=(4, 4)
    )

# Either run the full pipeline
plotter.run()

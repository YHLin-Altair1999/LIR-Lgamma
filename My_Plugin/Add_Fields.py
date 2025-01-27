import yt
import My_Plugin.Fields as f

def add_fields(ds):
    ds.add_field(   ('gas', 'volume'),
                    function = f.volume,
                    units = "cm**3",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'temp'),
                    function = f.temperature,
                    units = "K",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'Neutral_Hydrogen_Density'),
                    function = f.rho_HI,
                    units = "g*cm**(-3)",
                    sampling_type = "local"
                    )
    
    ds.add_field(   ('gas', 'Neutral_Hydrogen_Number_Density'),
                    function = f.n_HI,
                    units = "cm**(-3)",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'Internal_energy_density'),
                    function = f.e_int,
                    units = "eV/cm**3",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'CR_energy_density'),
                    function = f.e_cr,
                    units = "eV/cm**3",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'epsilon_gamma'),
                    function = f.epsilon_gamma,
                    units = "erg/(s*cm**3)",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'epsilon_gamma_incell'),
                    function = f.epsilon_gamma_incell,
                    units = "erg/s",
                    sampling_type = "local"
                    )

    #ds.add_field(   ('gas', 'metal_density'),
    #                function = f.metal_density,
    #                units = "g*cm**(-3)",
    #                sampling_type = "local"
    #                )
    ds.add_field(   ('gas', 'Compton_y'),
                    function = f.Compton_y,
                    units = "cm**(-1)",
                    sampling_type = "local"
                    )
    return ds

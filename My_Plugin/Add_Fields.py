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
    
    ds.add_field(   ('gas', 'u_B'),
                    function = f.u_B,
                    units = "erg/cm**3",
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

    ds.add_field(   ('gas', 'total_metallicity'),
                    function = f.total_metallicity,
                    units = "",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'Compton_y'),
                    function = f.Compton_y,
                    units = "cm**(-1)",
                    sampling_type = "local"
                    )
    
    ds.add_field(   ('gas', 'CRp_number_density'),
                    function = f.CRp_number_density,
                    units = "cm**(-3)",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'Pion_decay_gamma_ray_source_function'),
                    function = f.s_pi,
                    units = "cm**(-3)*GeV**(-1)*s**(-1)",
                    sampling_type = "local"
                    )

    ds.add_field(   ('gas', 'inverse_Compton_gamma_ray_source_function'),
                    function = f.s_IC,
                    units = "cm**(-3)*GeV**(-1)*s**(-1)",
                    sampling_type = "local"
                    )
    
    ds.add_field(   ('gas', 'Pion_decay_gamma_ray_source_function_in_cell'),
                    function = f.s_pi_incell,
                    units = "GeV**(-1)*s**(-1)",
                    sampling_type = "local"
                    )
                    
    ds.add_field(   ('gas', 'inverse_Compton_gamma_ray_source_function_in_cell'),
                    function = f.s_IC_incell,
                    units = "GeV**(-1)*s**(-1)",
                    sampling_type = "local"
                    )
                
    ds.add_field(   ('gas', 'gas_number_density'),
                    function = f.n_H,
                    units = "cm**(-3)",
                    sampling_type = "local",
                    display_name = r'$\rho/m_{\rm p}$'
                    )
    return ds

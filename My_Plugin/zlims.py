def get_zlim(field: tuple) -> list:
    '''
    Returns the zlim for a given field.
    The field is a tuple of (field_type, field_name).
    '''
    zlim_field_pairs = {
        ('gas', 'density'): [1e-4, 4e-1],
        ('gas', 'CR_energy_density'): [None, None],
        ('gas', 'Internal_energy_density'): [None, None],
        ('gas', 'epsilon_gamma'): [None, None],
        ('gas', 'metal_density'): [None, None],
        ('gas', 'total_metallicity'): [4e-4, 1],
        ('gas', 'Neutral_Hydrogen_Number_Density'): [None, None],
        ('gas', 'Compton_y'): [None, None],
        ('gas', 'gas_number_density'): [None, None],
    }
    return zlim_field_pairs.get(field, [None, None])





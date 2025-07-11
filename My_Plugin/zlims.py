def get_zlim(field: tuple) -> list:
    '''
    Returns the zlim for a given field.
    The field is a tuple of (field_type, field_name).
    '''
    zlim_field_pairs = {
        ('gas', 'density'): [1e-4, 4e-1],
        ('gas', 'CR_energy_density'): ['min', 'max'],
        ('gas', 'Internal_energy_density'): ['min', 'max'],
        ('gas', 'epsilon_gamma'): ['min', 'max'],
        ('gas', 'metal_density'): ['min', 'max'],
        ('gas', 'total_metallicity'): [4e-4, 1],
        ('gas', 'Neutral_Hydrogen_Number_Density'): ['min', 'max'],
        ('gas', 'Compton_y'): ['min', 'max'],
        ('gas', 'gas_number_density'): ['min', 'max'],
    }
    return zlim_field_pairs.get(field, ['min', 'max'])





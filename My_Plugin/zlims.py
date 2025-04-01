
def get_zlim(field: tuple) -> list:
    zlim_field_pairs = {
        ('gas', 'density'): [1e-4, 4e-1],
        ('gas', 'CR_energy_density'): [None, None],
        ('gas', 'Internal_energy_density'): [None, None],
        ('gas', 'epsilon_gamma'): [None, None],
        ('gas', 'metal_density'): [None, None],
        ('gas', 'Neutral_Hydrogen_Number_Density'): [None, None],
        ('gas', 'Compton_y'): [None, None],
    }
    return zlim_field_pairs.get(field)





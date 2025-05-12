import yt

def get_cmap(field):
    cmap_dict = {
        ('gas', 'density'): 'Spectral_r',
        ('gas', 'temp'): 'gist_heat',
        ('gas', 'Internal_energy_density'): 'gist_heat',
        ('gas', 'CR_energy_density'): 'inferno',
        ('gas', 'CRp_number_density'): 'inferno',
        ('gas', 'epsilon_gamma'): 'dusk',
        ('gas', 'Pion_decay_gamma_ray_source_function'): 'dusk',
        ('gas', 'inverse_Compton_gamma_ray_source_function'): 'dusk',
        }
    return cmap_dict.get(field, 'Spectral_r')

if __name__ == '__main__':
    field = ('gas', 'density')
    print(get_cmap(field))

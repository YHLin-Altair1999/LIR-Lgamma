import yt
import numpy as np

from yt import YTQuantity
import yt.utilities.physical_constants as c

code_length = YTQuantity(3.09e21, 'cm')
code_mass = YTQuantity(1.989e43, 'g')
code_velocity = YTQuantity(1e5, 'cm/s')

def volume(field, data):
    return data[('PartType0', 'Masses')] / data[('PartType0', 'Density')]

def e_int(field, data):
    e_int_sp = data[('PartType0', 'InternalEnergy')].v*code_velocity**2
    e_int = e_int_sp * data[('PartType0', 'Density')].v*code_mass/code_length**3
    return e_int

def temperature(field, data):
    # Column 7: temperature
    e_int_sp = data[('PartType0', 'InternalEnergy')].v*code_velocity**2
    helium_mass_fraction = data[('PartType0', 'Metallicity_01')]
    y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
    e_abundance = data[('PartType0','ElectronAbundance')]
    mu = (1 + 4*y_helium) / (1 + y_helium + e_abundance)
    mean_molecular_weight = mu*c.mp
    gamma = 5/3
    T = mean_molecular_weight * (gamma-1) * e_int_sp / c.kb
    return T

def e_cr(field, data):
    '''
    CR energy density, erg/cm**3
    '''
    e_cr = data[('PartType0', 'CosmicRayEnergy')].v*code_velocity**2*code_mass/data[('gas', 'volume')]
    return e_cr


def epsilon_gamma(field, data):
    '''
    GeV Gamma ray emissivity in units of erg/(s*cm**3).
    Ref: TK Chan et al. (2019) Eq. 6, 8
    https://arxiv.org/abs/1812.10496
    '''
    beta_pi = 0.7
    x_e = data[('PartType0', 'ElectronAbundance')]
    n_n = data[('PartType0', 'Density')] / c.mp 
    #f_Hydrogen = 1 - data[('PartType0', 'Metallicity_01')] - data[('PartType0', 'Metallicity_00')]
    #n_n = data[('PartType0', 'Density')] * f_Hydrogen / c.mp 
    e_cr = data[('gas', 'CR_energy_density')]
    Gamma_cr_had = (5.8e-16*
                    (e_cr/YTQuantity(1, "erg*cm**(-3)"))*
                    (n_n/YTQuantity(1, "cm**(-3)"))*
                    YTQuantity(1, 'erg*cm**(-3)*s**(-1)'))
    epsilon_cr_had = 1/3 * beta_pi * Gamma_cr_had
    return Gamma_cr_had

def epsilon_gamma_incell(field, data):
    return data[('gas', 'epsilon_gamma')]*data[('gas', 'volume')]

def metal_density(field, data):
    metal_density = data[('PartType0', 'Metallicity_00')]*data[('PartType0','Density')]
    return metal_density

def rho_HI(field, data):
    return data[('PartType0','NeutralHydrogenAbundance')]*data[('PartType0','Density')]

def n_HI(field, data):
    rho = data[('PartType0','Density')].v*code_mass*code_length**(-3)
    f_HI = data[('PartType0','NeutralHydrogenAbundance')]
    return rho * f_HI / c.mp

def Compton_y(field, data):
    x_e = data[('PartType0', 'ElectronAbundance')]
    f_Hydrogen = 1 - data[('PartType0', 'Metallicity_01')]
    n_n = data[('PartType0', 'Density')] * f_Hydrogen / c.mp 
    n_e = n_n*x_e
    y = c.sigma_thompson*c.kb*data[('gas', 'temperature')]*n_e/(c.me*c.c**2)
    return y 

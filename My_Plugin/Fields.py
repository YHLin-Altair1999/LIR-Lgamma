import yt
import numpy as np

from yt import YTQuantity
import yt.utilities.physical_constants as c
import yt.units as u
import scipy as sp

code_length = YTQuantity(3.09e21, 'cm')
code_mass = YTQuantity(1.989e43, 'g')
code_velocity = YTQuantity(1e5, 'cm/s')

def volume(field, data):
    return (data[('PartType0', 'Masses')].v*code_mass) / (data[('PartType0', 'Density')].v*code_mass/code_length**3)

def e_int(field, data):
    e_int_sp = data[('PartType0', 'InternalEnergy')].v*code_velocity**2
    e_int = e_int_sp * data[('PartType0', 'Density')].v*code_mass/code_length**3
    return e_int

def u_B(field, data):
    '''
    Magnetic energy density, erg/cm**3
    '''
    B_vec = data[('PartType0', 'MagneticField')].v * u.G
    B = np.linalg.norm(B_vec, axis=1)
    u_B  = B.v**2/(8*np.pi) * u.erg/u.cm**3
    return u_B.to('erg/cm**3')

def temperature(field, data):
    '''
    Calculates the temperature in K, using the suggested method from GIZMO
    Ref: http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html
    '''
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
    E_cr = data[('PartType0', 'CosmicRayEnergy')].v*code_velocity**2*code_mass
    if len(E_cr.shape) > 1:
        E_cr = np.sum(E_cr, axis=1) # sum over all CR energy
    e_cr = E_cr/data[('gas', 'volume')]
    return e_cr


def epsilon_gamma(field, data):
    '''
    GeV Gamma ray emissivity in units of erg/(s*cm**3).
    Ref: TK Chan et al. (2019) Eq. 6, 8
    https://arxiv.org/abs/1812.10496
    '''
    beta_pi = 0.7
    x_e = data[('PartType0', 'ElectronAbundance')]
    n_n = (data[('PartType0', 'Density')].v*code_mass/code_length**3) / c.mp
    e_cr = data[('gas', 'CR_energy_density')]
    Gamma_cr_had = (5.8e-16*
                    (e_cr/YTQuantity(1, "erg*cm**(-3)"))*
                    (n_n/YTQuantity(1, "cm**(-3)"))*
                    YTQuantity(1, 'erg*cm**(-3)*s**(-1)'))
    epsilon_cr_had = 1/3 * beta_pi * Gamma_cr_had
    return Gamma_cr_had

def CRp_number_density(field, data, alpha_p=2.2, q=0.5):
    e_cr = data[('gas', 'CR_energy_density')]
    rest_energy = c.mp * c.c**2
    energy_integral = 0.5 * \
        sp.special.betainc((alpha_p-2)/2, (3-alpha_p)/2, 1/(1+q**2)) * \
        sp.special.beta((alpha_p-2)/2, (3-alpha_p)/2) + \
        q**(alpha_p-1) * (np.sqrt(1+q**2) - 1)
    C_p = e_cr * (alpha_p - 1) / (rest_energy * energy_integral)
    return C_p

def s_pi(field, data, alpha_p=2.2, q=0.5, E=1*u.GeV):
    '''
    Calculate the gamma ray source function from neutral pion decay.
    '''
    C_p = data[('gas', 'CRp_number_density')]
    n_n = (data[('PartType0', 'Density')].v*code_mass/code_length**3) / c.mp

    delta = 0.14*alpha_p**(-1.6) + 0.44
    mbarn = 1e-27*u.cm**2
    sigma_pp = 32*(0.96 + np.exp(4.4 - 2.4*alpha_p))*mbarn
    m_pi = 134.9768*u.MeV/c.c**2 # neutral pion mass

    # Calculate s_pi: pion-decay gamma-ray luminosity
    energy_ratio = 2*E/(m_pi*c.c**2)
    energy_term = (energy_ratio**delta + energy_ratio**(-delta))**(-alpha_p/delta)
    mass_ratio = (c.mp/(2*m_pi))**alpha_p
    interaction_factor = sigma_pp*n_n/(c.mp*c.c)
    normalization = 16*C_p/(3*alpha_p)

    # Calculate pion-decay gamma-ray luminosity
    s_pi = normalization * interaction_factor * mass_ratio * energy_term
    return s_pi

def s_IC(field, data, alpha_p=2.2, q=0.5, E=1*u.GeV):
    '''
    Calculate the gamma ray source function from inverse Compton scattering.
    '''
    C_p = data[('gas', 'CRp_number_density')]
    n_n = (data[('PartType0', 'Density')].v*code_mass/code_length**3) / c.mp

    alpha_e = alpha_p + 1
    alpha_nu = (alpha_e - 1)/2
    sigma_pp = 32*(0.96 + np.exp(4.4-2.4*alpha_p))*(1e-27*u.cm**2)
    r_e = 2.8179403205e-15*u.m

    u_B = data[('gas', 'u_B')]
    u_CMB = 0.260*u.eV/u.cm**3
    
    C_e = 16**(2-alpha_e) * sigma_pp * n_n * C_p * c.me*c.c**2 / \
            ((alpha_e-2)*c.sigma_thompson*(u_B+u_CMB)) * (c.mp/c.me)**(alpha_e-2)
            
    f_IC = 2**(alpha_e+3)*(alpha_e**2 + 4*alpha_e + 11)/ \
            ((alpha_e+3)**2*(alpha_e+5)*(alpha_e+1)) * sp.special.gamma((alpha_e+5)/2) * sp.special.zeta((alpha_e+5)/2)
    
    s_IC = C_e*8*np.pi**2*r_e**2/(c.hcgs**3*c.c**2) * \
        (c.kb*c.Tcmb)**(3+alpha_nu) * f_IC * E**(-alpha_nu-1)
    return s_IC

def s_pi_incell(field, data):
    return data[('gas', 'Pion_decay_gamma_ray_source_function')]*data[('gas', 'volume')]

def s_IC_incell(field, data):
    return data[('gas', 'inverse_Compton_gamma_ray_source_function')]*data[('gas', 'volume')]

def epsilon_gamma_incell(field, data):
    return data[('gas', 'epsilon_gamma')]*data[('gas', 'volume')]

def metal_density(field, data):
    metal_density = data[('PartType0', 'Metallicity_00')]*data[('PartType0','Density')]
    return metal_density

def total_metallicity(field, data):
    return data[('PartType0', 'Metallicity_00')]

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

def n_H(field, data):
    return data[('gas', 'density')]/c.mp

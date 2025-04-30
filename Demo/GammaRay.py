import numpy as np
from matplotlib import rc
import scipy as sp
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from num2tex import num2tex

# TeX support
rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r'\usepackage{amsmath}')

class One_Sim:
    def __init__(self, n_n=1e0*u.cm**-3, e_cr=1.0e-11*u.erg/u.cm**3, B=0*u.G, alpha_p=2.2, 
                 E_1=1e-3*u.GeV, E_2=1e3*u.GeV, q=0.5):
        """Initialize simulation parameters"""
        self.n_n = n_n  # thermal nucleon number density
        self.e_cr = e_cr  # CR energy density
        self.B = B # magnetic field strength
        self.alpha_p = alpha_p  # CR proton spectral index
        self.E_1 = E_1  # Lower energy bound
        self.E_2 = E_2  # Upper energy bound
        self.q = q  # momentum cutoff parameter

        # Constants
        self.u_CMB = 0.260*u.eV/u.cm**3
        self.u_B = (B.to('Gauss').value)**2/(8*np.pi)*u.erg/u.cm**3
        self.T_CMB = 2.726*u.K
        self.m_pi = 134.9768*u.MeV/c.c**2
        self.r_e = 2.8179403205e-15*u.m
    
    def C_p(self, alpha_p, q):
        """Calculate C_p for CR protons"""

        rest_energy = c.m_p * c.c**2
        energy_integral = 0.5 * \
            sp.special.betainc((alpha_p-2)/2, (3-alpha_p)/2, 1/(1+q**2)) * \
            sp.special.beta((alpha_p-2)/2, (3-alpha_p)/2) + \
            q**(alpha_p-1) * (np.sqrt(1+q**2) - 1)
        C_p = self.e_cr * (alpha_p - 1) / (rest_energy * energy_integral)
        return C_p

    def calculate_s_pi(self, E, alpha_p=None, q=None):
        """Calculate pion-decay gamma-ray luminosity"""
        # Use instance values if not provided
        alpha_p = alpha_p if alpha_p is not None else self.alpha_p
        q = q if q is not None else self.q

        C_p = self.C_p(alpha_p=alpha_p, q=q)
        
        delta = 0.14*alpha_p**(-1.6) + 0.44
        sigma_pp = 32*(0.96 + np.exp(4.4 - 2.4*alpha_p))*u.mbarn
        
        # Calculate s_pi: pion-decay gamma-ray luminosity
        energy_ratio = 2*E/(self.m_pi*c.c**2)
        energy_term = (energy_ratio**delta + energy_ratio**(-delta))**(-alpha_p/delta)
        mass_ratio = (c.m_p/(2*self.m_pi))**alpha_p
        interaction_factor = sigma_pp*self.n_n/(c.m_p*c.c)
        normalization = 16*C_p/(3*alpha_p)

        # Calculate pion-decay gamma-ray luminosity
        s_pi = normalization * interaction_factor * mass_ratio * energy_term
        return s_pi.to(u.s**-1*u.cm**-3*u.GeV**-1)

    def calculate_s_IC(self, E, alpha_p=None, q=None):
        """Calculate inverse Compton gamma-ray luminosity"""
        # Use instance values if not provided
        alpha_p = alpha_p if alpha_p is not None else self.alpha_p
        q = q if q is not None else self.q

        alpha_e = alpha_p + 1
        alpha_nu = (alpha_e - 1)/2
        sigma_pp = 32*(0.96 + np.exp(4.4-2.4*alpha_p))*u.mbarn
        
        C_e = 16**(2-alpha_e) * sigma_pp * self.n_n * self.C_p(alpha_p=alpha_p, q=q) * c.m_e*c.c**2 / \
              ((alpha_e-2)*c.sigma_T*(self.u_B+self.u_CMB)) * (c.m_p/c.m_e)**(alpha_e-2)
               
        f_IC = 2**(alpha_e+3)*(alpha_e**2 + 4*alpha_e + 11)/ \
               ((alpha_e+3)**2*(alpha_e+5)*(alpha_e+1)) * sp.special.gamma((alpha_e+5)/2) * sp.special.zeta((alpha_e+5)/2)
        
        s_IC = C_e*8*np.pi**2*self.r_e**2/(c.h**3*c.c**2) * \
            (c.k_B*self.T_CMB)**(3+alpha_nu) * f_IC * E**(-alpha_nu-1)
        
        return s_IC.to(u.s**-1*u.cm**-3*u.GeV**-1)

    def calculate_Lambda_pi(self, E_range, alpha_p=None, q=None):
        s_pi = self.calculate_s_pi(E=E_range, alpha_p=alpha_p, q=q)
        Lambda_pi = sp.integrate.simpson((s_pi*E_range).value, x=E_range.value) * E_range.unit**2 * s_pi.unit
        return Lambda_pi.to(u.erg*u.s**-1*u.cm**-3)
    
    def calculate_Lambda_IC(self, E_range, alpha_p=None, q=None):
        s_IC = self.calculate_s_IC(E=E_range, alpha_p=alpha_p, q=q)
        Lambda_IC = sp.integrate.simpson((s_IC*E_range).value, x=E_range.value) * E_range.unit**2 * s_IC.unit
        return Lambda_IC.to(u.erg*u.s**-1*u.cm**-3)

    def plot_spectrum(self, E_range=None, alpha_p_values=None, save=True):
        """Plot gamma-ray spectrum for different alpha_p values"""
        if E_range is None:
            E_range = np.logspace(np.log10(self.E_1.to('GeV').value), np.log10(self.E_2.to('GeV').value), 1000)*u.GeV
        if alpha_p_values is None:
            alpha_p_values = [2.05, 2.30, 2.55, 2.80]

        fig, ax = plt.subplots(figsize=(6,4))
        xunit = E_range.unit
        yunit = u.s**-1*u.cm**-3*u.GeV**-1

        # Set up the colormap
        cmap = plt.cm.Blues
        norm = plt.Normalize(1, max(alpha_p_values))
        
        # Create empty line objects for legend
        legend_lines = []
        legend_labels = []
        
        # Add style legend entries
        legend_lines.append(plt.Line2D([0], [0], color='gray', linestyle='-'))
        legend_lines.append(plt.Line2D([0], [0], color='gray', linestyle='--'))
        legend_labels.append(r'$\pi^0$ decay')
        legend_labels.append(r'inverse Compton')
        
        for idx, alpha_p in enumerate(alpha_p_values):
            color = cmap(norm(alpha_p))

            s_pi = self.calculate_s_pi(E=E_range, alpha_p=alpha_p, q=self.q)#*E_range
            s_IC = self.calculate_s_IC(E=E_range, alpha_p=alpha_p, q=self.q)#*E_range
            # one can multiply by E_range to get the luminosity spectrum
            # but remember to change yunit when plotting

            # Plot Lambda_pi (solid) and Lambda_IC (dashed) without labels
            ax.loglog(E_range.to(xunit).value, 
                 s_pi.to(yunit).value, 
                 color=color, linestyle='-')
            ax.loglog(E_range.to(xunit).value, 
                 s_IC.to(yunit).value, 
                 color=color, linestyle='--')

            # Add color legend entries
            legend_lines.append(plt.Line2D([0], [0], color=color))
            legend_labels.append(f'$\\alpha_p = {alpha_p:.2f}$')

        # Add text showing q value at bottom left corner
        ax.text(0.05, 0.05, f'$q = {self.q}$', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0., edgecolor='none'), 
            fontsize=9, ha='left', va='bottom')
        ax.text(0.05, 0.1, f'$n_n = {num2tex(self.n_n.value)}$ {self.n_n.unit.to_string("latex_inline")}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0., edgecolor='none'), 
            fontsize=9, ha='left', va='bottom')
        ax.text(0.05, 0.15, f'$e_{{\\rm cr}} = {num2tex(self.e_cr.value)}$ {self.e_cr.unit.to_string("latex_inline")}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0., edgecolor='none'), 
            fontsize=9, ha='left', va='bottom')
        ax.text(0.05, 0.2, f'$|B| = {num2tex(self.B.value)}$ {self.B.unit.to_string("latex_inline")}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0., edgecolor='none'), 
            fontsize=9, ha='left', va='bottom')
        ax.set_xlabel(r'$E_\gamma$ (GeV)')
        ax.set_ylabel(rf'$s_\gamma$ {{\rm (ph~{yunit.to_string("latex_inline")}}})')
        ax.set_ylim(1e-31,1e-19)
        ax.set_xlim(E_range[0].to(xunit).value, E_range[-1].to(xunit).value)
        ax.legend(legend_lines, legend_labels, loc='best')
        ax.grid(True, linestyle='-', linewidth=0.5)
        plt.tight_layout()
        fig.savefig('s_gamma_spectrum.png', dpi=300)
        return None
    
    def calculate_luminosity_Pfrommer(self, E_1, E_2, alpha_p, q):
        """Calculate luminosity from spectrum"""
        Eunit = E_1.unit
        E_range = np.logspace(np.log10(E_1.to(Eunit).value), np.log10(E_2.to(Eunit).value), 1000)*Eunit
        L_pi = self.calculate_Lambda_pi(E_range, alpha_p=alpha_p, q=q)
        L_IC = self.calculate_Lambda_IC(E_range, alpha_p=alpha_p, q=q)
        return [L_pi + L_IC, L_pi, L_IC]
    
    def calculate_luminosity_Chan(self):
        '''Calculate luminosity from Chan 2019, which comes from Guo and Oh 2008'''
        beta_pi = 0.7
        return 1/3 * beta_pi * 5.8e-16 * self.e_cr.to('erg*cm**-3').value * self.n_n.to('cm**-3').value * u.erg*u.s**-1*u.cm**-3
    
    def compare_luminosity(self, E_1, E_2, alpha_p, q):
        """Compare luminosity from Pfrommer and Chan"""
        L_Pfrommer = self.calculate_luminosity_Pfrommer(E_1, E_2, alpha_p, q)[0]
        L_Chan = self.calculate_luminosity_Chan()
        print(f'Under the assumptions of alpha_p = {alpha_p}, q = {q}, E_1 = {E_1}, E_2 = {E_2}')
        print(f'Pfrommer 2017 gives emissivity: {L_Pfrommer.to("erg*s**-1*cm**-3")}')
        print(f'Chan 2019 gives emissivity: {L_Chan.to("erg*s**-1*cm**-3")}')
        print(f'Ratio: {(L_Pfrommer/L_Chan).to('')}')
        return L_Pfrommer, L_Chan
    
    def plot_integral_luminosity(self, E_range=None, alpha_p_values=None, E_max=1e6*u.PeV, save=True):
        """Plot integrated gamma-ray luminosity from E to E_max for different alpha_p values"""
        if E_range is None:
            E_range = np.logspace(-3, 3, 50)*u.GeV  # Points for plotting
        if alpha_p_values is None:
            alpha_p_values = [2.05, 2.30, 2.55, 2.80]
        
        # Create high-resolution integration range
        integration_range = np.logspace(
            np.log10(E_range[0].to('GeV').value), 
            np.log10(E_max.to('GeV').value), 
            1000  # High resolution for accurate integration
        )*u.GeV
        
        fig, ax = plt.subplots(figsize=(6,4))
        xunit = u.GeV
        yunit = u.erg*u.s**-1*u.cm**-3
        
        # Set up colormap
        cmap = plt.cm.Blues
        norm = plt.Normalize(1, max(alpha_p_values))
        
        # Create legend elements
        legend_lines = []
        legend_labels = []
        
        # Add style legend entries
        legend_lines.append(plt.Line2D([0], [0], color='gray', linestyle='-'))
        legend_lines.append(plt.Line2D([0], [0], color='gray', linestyle='--'))
        legend_labels.append(r'$\Lambda_\pi(>E_\gamma)$')
        legend_labels.append(r'$\Lambda_{\rm IC}(>E_\gamma)$')
        
        for idx, alpha_p in enumerate(alpha_p_values):
            color = cmap(norm(alpha_p))
            
            # Calculate spectrum over full integration range
            s_pi = self.calculate_s_pi(E=integration_range, alpha_p=alpha_p, q=self.q)
            s_IC = self.calculate_s_IC(E=integration_range, alpha_p=alpha_p, q=self.q)
            
            # Initialize result arrays
            L_pi = np.zeros(len(E_range))*yunit
            L_IC = np.zeros(len(E_range))*yunit
            
            # For each point in E_range, calculate luminosity from that point to E_max
            for i, E_min in enumerate(E_range):
                # Find starting index for integration
                idx_start = np.argmin(np.abs(integration_range - E_min))
                
                # Integrate from E_min to E_max
                L_pi[i] = sp.integrate.simpson(
                    (s_pi[idx_start:]*integration_range[idx_start:]).value, 
                    x=integration_range[idx_start:].value
                ) * integration_range.unit**2 * s_pi.unit
                
                L_IC[i] = sp.integrate.simpson(
                    (s_IC[idx_start:]*integration_range[idx_start:]).value, 
                    x=integration_range[idx_start:].value
                ) * integration_range.unit**2 * s_IC.unit
            
            # Plot Lambda_pi and Lambda_IC
            ax.loglog(E_range.to(xunit).value, 
                    L_pi.to(yunit).value, 
                    color=color, linestyle='-')
            ax.loglog(E_range.to(xunit).value, 
                    L_IC.to(yunit).value, 
                    color=color, linestyle='--')
            
            # Add color legend entries
            legend_lines.append(plt.Line2D([0], [0], color=color))
            legend_labels.append(f'$\\alpha_p = {alpha_p:.2f}$')
        
        # Add parameter annotations
        ax.text(0.05, 0.05, f'$q = {self.q}$', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0., edgecolor='none'), 
            fontsize=9, ha='left', va='bottom')
        ax.text(0.05, 0.1, f'$n_n = {num2tex(self.n_n.value)}$ {self.n_n.unit.to_string("latex_inline")}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0., edgecolor='none'), 
            fontsize=9, ha='left', va='bottom')
        #ax.text(0.05, 0.15, f'$E_{{\\rm max}} = 10^6$ PeV', transform=ax.transAxes, 
        #    bbox=dict(facecolor='white', alpha=0., edgecolor='none'), 
        #    fontsize=9, ha='left', va='bottom')
        ax.text(0.05, 0.15, f'$e_{{\\rm cr}} = {num2tex(self.e_cr.value)}$ {self.e_cr.unit.to_string("latex_inline")}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0., edgecolor='none'), 
            fontsize=9, ha='left', va='bottom')
        
        ax.set_xlabel(r'$E_\gamma$ (GeV)')
        ax.set_ylabel(rf'$\Lambda_\gamma(>E_\gamma)$ {{\rm ({yunit.to_string("latex_inline")}}})')
        ax.set_xlim(E_range[0].to(xunit).value, E_range[-1].to(xunit).value)
        ax.set_ylim(1e-31, 1e-26)
        ax.grid(True, linestyle='-', linewidth=0.5)
        #ax.legend(legend_lines, legend_labels, loc='best')
        
        plt.tight_layout()
        if save:
            fig.savefig('L_gamma_integral_spectrum.png', dpi=300)
        
        return fig, ax

if __name__ == '__main__':
    sim = One_Sim()
    sim.compare_luminosity(1*u.GeV, 1e6*u.GeV, 2.2, 0.5)
    sim.plot_spectrum()
    sim.plot_integral_luminosity(E_range=np.logspace(-3, 3, 50)*u.GeV, alpha_p_values=[2.05, 2.30, 2.55, 2.80])
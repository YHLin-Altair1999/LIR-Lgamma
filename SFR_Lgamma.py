import numpy as np
import pandas as pd
from LIR_Lgamma import LIR_Lgamma_Plot
import astropy.units as u
import matplotlib.pyplot as plt
import os
from astropy.cosmology import Planck18, z_at_value

class SFR_Lgamma_Plot(LIR_Lgamma_Plot):
    def __init__(self, 
                galaxies=None,
                E_min=1*u.GeV, E_max=1000*u.GeV,
                Lgamma_profile_folder='/tscc/lustre/ddn/scratch/yel051/tables/Lgamma_profiles',
                SFR_profile_folder='/tscc/lustre/ddn/scratch/yel051/tables/SFR_profiles',
                sim_table_path='./tables/Lgamma_SFR.csv',
                show_obs_gal_name=False,
                aperture=25*u.kpc):
        """Initialize SFR_Lgamma_Plot with parameters for analysis and visualization"""
        super().__init__(
            galaxies=galaxies,
            E_min=E_min, 
            E_max=E_max,
            Lgamma_profile_folder=Lgamma_profile_folder,
            sim_table_path=sim_table_path,
            show_obs_gal_name=show_obs_gal_name
        )
        self.SFR_profile_folder = SFR_profile_folder
        self.aperture = aperture
        self.x_range = np.logspace(-3, 3, 100)  # SFR range differs from LIR
    
    def make_sim_table(self):
        """Create or update table with gamma ray luminosities and SFR"""
        data = []
        
        for galaxy, snaps in self.galaxies.items():
            for snap in snaps:
                fname = os.path.join(self.Lgamma_profile_folder, f'Lgamma_profile_{galaxy}_snap{snap:03d}.npy')
                profile = np.load(fname)
                Lgamma = np.sum(profile[:,1])
                
                fname = os.path.join(self.SFR_profile_folder, f'SFR_profile_{galaxy}_snap{snap:03d}.npy')
                profile = np.load(fname)
                SFR = np.sum(profile[:,1])*u.Msun/u.yr
                
                data.append({
                    'galaxy': galaxy, 
                    'snap': snap, 
                    'L_gamma (erg/s)': Lgamma,
                    'SFR (M_sun/yr)': SFR.to('Msun/yr').value
                })

        df = pd.DataFrame(data)
        df = df.sort_values(by='snap')
        df.to_csv(self.sim_table_path, index=False)
        self.sim_table = df
    
    def load_obs_data(self):
        """Load observational data and convert LIR to SFR"""
        obs_path = './obs_data/Ambrosone_2024.csv'
        df = pd.read_csv(obs_path)
        df['L_IR (L_sun)'] = np.array(df['LIR'])*1e10
        df['z'] = z_at_value(self.cosmo.luminosity_distance, np.array(df['DL'])*u.Mpc)
        
        # Calculate L_gamma values
        L_gammas = np.array([
            self.luminosity(
                np.logspace(np.log10(self.E_min.to(u.GeV).value),np.log10(self.E_max.to(u.GeV).value),100) * u.GeV, 
                df['Phi0'][i] * 1e-12 * u.cm**-2 * u.s**-1 * u.MeV**-1, 
                df['gamma'][i], 
                df['DL'][i]*u.Mpc
            ) for i in range(len(df['z']))
        ])
        
        # Convert LIR to SFR
        epsilon = 0.79  # See Pfrommer 2017 Eq. 15
        df['SFR (M_sun/yr)'] = epsilon*1.7e-10*df['L_IR (L_sun)']
        
        # Store all observational data
        self.obs_data = {
            'df': df,
            'L_gammas': L_gammas,
        }
    
    def fit_obs_data(self):
        """Fit power law to observational data"""
        df = self.obs_data['df']
        L_gammas = self.obs_data['L_gammas']
        
        # Remove specific galaxies from fitting
        galaxies_to_exclude = ['Circinus Galaxy', 'NGC 2403', 'NGC 3424']
        fitting_mask = ~df['Source'].isin(galaxies_to_exclude)
        
        # Store L_gamma values corresponding to filtered galaxies
        L_gammas_filtered = L_gammas[fitting_mask]
        
        # Create filtered dataframe with L_gamma values
        df_for_fitting = df[fitting_mask].copy()
        df_for_fitting['L_gamma'] = L_gammas_filtered
        
        # Fit power law
        self.obs_fit_result = self._fit_powerlaw(
            x_data=df_for_fitting['SFR (M_sun/yr)'],
            y_data=df_for_fitting['L_gamma'],
            label_prefix='Obs'
        )
        
        # Store x and y data for residual calculation
        self.obs_fit_result['data_x'] = df['SFR (M_sun/yr)']
        self.obs_fit_result['data_y'] = L_gammas
        self.obs_fit_function = self.obs_fit_result['fit_function']
    
    def plot_main(self, ax=None, sim_table_path=None):
        """Plot both simulation and observational data on the main panel"""
        sim_table_path = sim_table_path or self.sim_table_path
        ax = ax or self.axes[0]
        
        # Plot simulation data
        df = pd.read_csv(sim_table_path)
        for i in range(df.shape[0]):
            galaxy = df['galaxy'][i]
            ax.scatter(
                df['SFR (M_sun/yr)'][i], df['L_gamma (erg/s)'][i], 
                marker=self.get_marker(galaxy=galaxy), color=self.get_color(galaxy), 
                zorder=3, s=60, alpha=0.7, edgecolor='None')
        
        # Fit power law for simulation data
        self.sim_fit_result = self._fit_powerlaw(
            x_data=df['SFR (M_sun/yr)'],
            y_data=df['L_gamma (erg/s)'],
            label_prefix='Sim'
        )
        
        # Plot simulation fit if enabled
        if self.plot_sim_fit_main:
            self._plot_fit_result(ax, self.sim_fit_result, color='skyblue', label='Simulation fit')
        
        # Plot observational data
        if self.obs_data is None:
            print("Warning: No observational data loaded for plotting")
        else:
            df_obs = self.obs_data['df']
            L_gammas = self.obs_data['L_gammas']
            
            # Plot the data points
            ax.scatter(
                df_obs['SFR (M_sun/yr)'], L_gammas,
                label='Ambrosone et al. (2024)', color='gray', marker='^', 
                edgecolor='None', alpha=0.7, s=60, zorder=2
            )
            
            # Add galaxy names next to data points if requested
            if self.show_obs_gal_name:
                for i in range(len(L_gammas)):
                    ax.annotate(
                        df_obs['Source'][i], 
                        (df_obs['SFR (M_sun/yr)'][i], L_gammas[i]),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7
                    )
            
            # Plot the observation fit line and confidence band if enabled
            if self.plot_obs_fit_main and self.obs_fit_result:
                self._plot_fit_result(ax, self.obs_fit_result, color='gray', 
                                     label='Observation fit')
        
        # Configure the main axis
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        ax.set_xlim(self.x_range[0], self.x_range[-1])
        ax.legend()
    
    def plot_residuals(self, ax):
        """Plot residuals (data/best fit) in the lower panel using a log scale"""
        # Skip residual plotting if both controls are disabled
        if not (self.plot_obs_fit_residuals or self.plot_sim_residuals):
            ax.set_visible(False)
            return
            
        # Track all ratios to set y-limits later
        all_ratios = []
        
        # Plot simulation residuals relative to the observed fit if enabled
        if self.plot_sim_residuals:
            x_data = self.sim_table['SFR (M_sun/yr)']
            y_data = self.sim_table['L_gamma (erg/s)']
            # Calculate model values using the observed fit
            y_model = self.obs_fit_function(x_data)
            # Calculate ratios (data/model)
            ratios = y_data/y_model
            all_ratios.extend(ratios)
            
            # Plot ratios for each point
            for i, (x, ratio) in enumerate(zip(x_data, ratios)):
                galaxy = list(self.galaxies.keys())[i] if i < len(self.galaxies) else None
                ax.scatter(
                    x, ratio,
                    marker=self.get_marker(galaxy=galaxy) if galaxy else 'o', 
                    color=self.get_color(galaxy) if galaxy else 'blue', 
                    zorder=3, s=60, alpha=0.7, 
                    edgecolor='None'
                )
        
        # Plot observation residuals if enabled
        if self.plot_obs_fit_residuals and self.obs_fit_result:
            x_data = self.obs_fit_result['data_x']
            y_data = self.obs_fit_result['data_y']
            # Calculate model values using the observed fit
            y_model = self.obs_fit_function(x_data)
            # Calculate ratios (data/model)
            ratios = y_data/y_model
            all_ratios.extend(ratios)
            
            ax.scatter(
                x_data, ratios,
                color='gray', marker='^', edgecolor='None', alpha=0.7, s=60, zorder=2
            )
        
        # Add confidence region from the observation fit
        if self.plot_obs_fit_residuals and self.obs_fit_result and 'alpha_error' in self.obs_fit_result:
            # Calculate confidence band for display
            ref_point = self.obs_fit_result['ref_point']
            alpha_error = self.obs_fit_result['alpha_error']
            beta_error = self.obs_fit_result['beta_error']
            
            # Create the confidence band for ratios
            x_range = np.logspace(np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1]), 100)
            log_x_range = np.log10(x_range/ref_point)
            
            # Calculate upper and lower bounds
            log_y_upper = np.maximum(
                alpha_error * log_x_range + beta_error,
                -alpha_error * log_x_range + beta_error
            )
            log_y_lower = np.minimum(
                -alpha_error * log_x_range - beta_error,
                alpha_error * log_x_range - beta_error
            )
            
            # Convert to linear space
            y_upper = 10**log_y_upper
            y_lower = 10**log_y_lower
            
            # Plot the confidence band
            ax.fill_between(x_range, y_lower, y_upper, color='gray', alpha=0.1, zorder=1)
            
            # Add a horizontal line at y=1
            ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, zorder=1, alpha=0.1)
        
        # Configure the axis
        ax.set_yscale('log')
        if all_ratios:
            ratio = 10**(1.2*np.max(np.abs(np.log10(np.array(all_ratios)))))
            ax.set_ylim(1/ratio, ratio)
        ax.set_ylabel(r'$L_{\gamma}/L_{\gamma,\mathrm{obs-fit}}$')
    
    def finalize(self, fig, axes):
        """Finalize plot with labels, scales, and save"""
        # Configure upper panel (with data)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        axes[0].set_xlim(self.x_range[0], self.x_range[-1])
        axes[0].legend()
        
        # Configure lower panel (with residuals)
        axes[1].set_xscale('log')
        axes[1].set_xlabel(r'$\mathrm{SFR}~(M_\odot/\mathrm{yr})$')
        axes[1].set_xlim(self.x_range[0], self.x_range[-1])
        
        plt.tight_layout()
        fig.savefig('SFR-Lgamma.png', dpi=300)

def main(E_min=1*u.GeV, E_max=1000*u.GeV, show_names=False):
    print('Running SFR_Lgamma.py')
    galaxies = {
        'm12i_et': [60], 
        'm12i_sc_fx10': [60], 
        'm12i_sc_fx100': [60],
        'm12i_cd': [600],
        'm11b_cd': [600],
        'm11c_cd': [600],
        'm11d_cd': [600],
        'm11f_cd': [600],
        'm11g_cd': [600],
        'm11h_cd': [600],
        'm11v_cd': [600],
        #'m10v_cd': [600],
        'm11f_et_AlfvenMax': [600],
        'm11f_et_FastMax': [600],
        'm11f_sc_fcas50': [600]
    }
    
    plotter = SFR_Lgamma_Plot(
        galaxies=galaxies,
        E_min=E_min, 
        E_max=E_max,
        show_obs_gal_name=show_names,
        aperture=25*u.kpc
    )
    
    plotter.run()

if __name__ == '__main__':
    main(E_min=1*u.GeV, E_max=1000*u.GeV, show_names=False)
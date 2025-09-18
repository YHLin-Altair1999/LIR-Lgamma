import numpy as np
import pandas as pd
from My_Plugin.LIR_Lgamma import LIR_Lgamma_Plot
import astropy.units as u
import matplotlib.pyplot as plt
import os
import pickle
from astropy.cosmology import Planck18, z_at_value

class SFR_Lgamma_Plot(LIR_Lgamma_Plot):
    def __init__(self, 
                galaxies=None,
                E_min=1*u.GeV, E_max=1000*u.GeV,
                Lgamma_profile_folder='/tscc/lustre/ddn/scratch/yel051/tables/Lgamma_profiles',
                SFR_folder='/tscc/lustre/ddn/scratch/yel051/tables/SFR',
                sim_table_path='./tables/Lgamma_SFR.csv',
                obs_pickle_path='./obs_data/fit_results.pkl',  # Use pickle file for observations
                show_obs_gal_name=False,
                output_filename='SFR_Lgamma.png',
                aperture=25*u.kpc):
        """Initialize SFR_Lgamma_Plot with parameters for analysis and visualization"""
        super().__init__(
            galaxies=galaxies,
            E_min=E_min, 
            E_max=E_max,
            Lgamma_profile_folder=Lgamma_profile_folder,
            sim_table_path=sim_table_path,
            obs_pickle_path=obs_pickle_path,  # Pass pickle path to parent
            show_obs_gal_name=show_obs_gal_name,
            output_filename=output_filename
        )
        self.SFR_folder = SFR_folder
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
                
                fname = os.path.join(self.SFR_folder, f'SFR_{galaxy}_snap{snap:03d}.npy')
                SFR, SFR_err = np.load(fname)*u.Msun/u.yr
                
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
        """Load observational data from pickle and convert LIR to SFR"""
        # Use parent class method to load from pickle
        super().load_obs_data()
        
        # Convert LIR to SFR for the loaded data
        if self.obs_data and 'df' in self.obs_data:
            df = self.obs_data['df']
            # Convert LIR to SFR using the same conversion as before
            epsilon = 0.79  # See Pfrommer 2017 Eq. 15
            df['SFR (M_sun/yr)'] = epsilon * 1.7e-10 * df['L_IR (L_sun)']
            
            # Update the stored data
            self.obs_data['df'] = df
    
    def fit_obs_data(self):
        """Load observational fit from pickle and adapt for SFR"""
        # Use parent class method to load fit from pickle
        super().fit_obs_data()
        
        # The parent method loads the fit in terms of LIR
        # We need to adapt the fit function to work with SFR
        if hasattr(self, 'obs_fit_function'):
            # Store the original LIR-based fit function
            original_fit_function = self.obs_fit_function
            
            # Create a new fit function that converts SFR to LIR first
            def sfr_fit_function(sfr_values):
                # Convert SFR to LIR using inverse of the conversion
                # SFR = epsilon * 1.7e-10 * LIR
                # So LIR = SFR / (epsilon * 1.7e-10)
                epsilon = 0.79
                lir_values = sfr_values / (epsilon * 1.7e-10)
                return original_fit_function(lir_values)
            
            # Replace the fit function
            self.obs_fit_function = sfr_fit_function
            
            # Update data arrays to use SFR instead of LIR
            if self.obs_fit_result and 'data_x' in self.obs_fit_result:
                # Convert LIR data to SFR data
                epsilon = 0.79
                self.obs_fit_result['data_x'] = self.obs_fit_result['data_x'] * epsilon * 1.7e-10
    
    def plot_obs_from_pickle(self, ax, pickle_file='fit_results.pkl', interval_type='prediction'):
        """Plot observational data using SFR instead of LIR"""
        # Use parent method but adapt for SFR plotting
        # Load results from pickle
        with open(pickle_file, 'rb') as f:
            results = pickle.load(f)
        
        # Extract data
        fit_params = results['fit_parameters']
        data = results['data']
        fit_result = results['fit_result']
        df = results['dataframe']
        fit_info = results['fit_info']
        
        # Use provided interval_type or the one from the saved results
        if interval_type is None:
            interval_type = fit_info['interval_type']
        
        # Determine which points were included/excluded
        excluded_galaxies = data['excluded_galaxies']
        source_names = data['source_names']
        included_mask = ~np.isin(source_names, excluded_galaxies)
        
        # Convert LIR to SFR for x-axis data
        epsilon = 0.79
        x_data_lir = data['L_IR_Lsun']
        x_data = x_data_lir * epsilon * 1.7e-10  # Convert to SFR
        y_data = data['L_gamma_erg_per_s']
        
        # Get asymmetric errors from dataframe for proper error bars
        y_err_minus = df['L_gamma_err_minus'].values
        y_err_plus = df['L_gamma_err_plus'].values
        yerr = [y_err_minus, y_err_plus]  # Format for matplotlib errorbar
        
        # Plot included points with filled triangles
        ax.scatter(
            x_data[included_mask], y_data[included_mask],
            marker='^', color='gray', edgecolor='None',
            s=49, alpha=0.7, zorder=3,
            label='Ambrosone et al. (2024)'
        )
        
        # Plot excluded points with empty triangles
        if np.any(~included_mask):
            ax.scatter(
                x_data[~included_mask], y_data[~included_mask],
                facecolors='none', marker='^', 
                edgecolor='gray', alpha=0.7, s=49, zorder=2
            )
        
        # Plot error bars for all points if enabled
        if self.show_obs_errorbars:
            ax.errorbar(
                x_data, y_data, 
                yerr=[yerr[0], yerr[1]],
                fmt='None',
                ecolor='gray',
                capsize=3,
                alpha=0.7,
                zorder=2
            )
        
        # Plot fit line and confidence interval (convert LIR range to SFR range)
        x_fit_lir = self.x_range / (epsilon * 1.7e-10)  # Convert SFR range back to LIR for fitting
        x_fit = self.x_range  # SFR range for plotting
        A = fit_params['A']
        B = fit_params['B']
        y_fit = A * x_fit_lir**B  # Calculate using LIR-based fit
        
        # Calculate confidence/prediction interval using the saved parameters
        if 'confidence_function' in fit_result:
            y_lower, y_upper = fit_result['confidence_function'](x_fit_lir, interval_type)
        elif 'confidence_params' in fit_result:
            # Recreate confidence intervals from stored parameters
            params = fit_result['confidence_params']
            log_x_plot = np.log10(x_fit_lir)
            
            if interval_type == 'confidence':
                se = params['residual_std'] * np.sqrt(1/params['n_points'] + (log_x_plot - params['x_mean'])**2 / params['sum_x_sq'])
            else:  # prediction interval
                se = params['residual_std'] * np.sqrt(1 + 1/params['n_points'] + (log_x_plot - params['x_mean'])**2 / params['sum_x_sq'])
            
            log_y_plot = params['actual_intercept'] + params['slope'] * log_x_plot
            y_upper = 10**(log_y_plot + se)
            y_lower = 10**(log_y_plot - se)
        else:
            # Fallback method
            y_upper = y_fit * 1.3
            y_lower = y_fit * 0.7
        
        # Plot best fit line
        fit_label = r'Best fit: $L_\gamma = {:.2e} \cdot \mathrm{{SFR}}^{{{:.2f}}}$'.format(A/(epsilon * 1.7e-10)**B, B)
        ax.plot(x_fit, y_fit, '--', color='gray', zorder=4, alpha=0.5)
        
        # Plot confidence/prediction interval
        interval_label = f'$1\\sigma$ {interval_type} interval'
        ax.fill_between(x_fit, y_lower, y_upper, color='gray', alpha=0.2, zorder=2)
        
        # Add galaxy names next to data points if requested
        if self.show_obs_gal_name:
            for i in range(len(y_data)):
                ax.annotate(
                    df['Source'].iloc[i], 
                    (x_data[i], y_data[i]),
                    xytext=(5, 0), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
        
        # Store the fit result for residual plotting with proper structure
        # Convert to the format expected by the parent class if needed
        epsilon = 0.79
        alpha = B  # power law exponent
        beta = np.log10(A / (epsilon * 1.7e-10)**B)  # Convert to SFR-based normalization
        
        self.obs_fit_result = {
            'A': A,
            'B': B,
            'alpha': alpha,
            'alpha_error': 0.0,  # We don't have error info in this context
            'beta': beta,
            'beta_error': 0.0,   # We don't have error info in this context
            'ref_point': 1.0,
            'fit_function': lambda x: A * (x/(epsilon * 1.7e-10))**B,  # Convert SFR to LIR internally
            'data_x': x_data,
            'data_y': y_data
        }
        self.obs_fit_function = self.obs_fit_result['fit_function']
        
        print(f"Loaded observational data and fit from {pickle_file}")
        return True

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
        
        # Plot observational data using pickle file method
        self.plot_obs_from_pickle(ax, self.obs_pickle_path, interval_type='prediction')
        
        # Plot calorimetric limit
        # Using equ. 10 from Chan et al. (2019)
        calorimetric_limit_sfr = self.x_range * u.Msun/u.yr
        calorimetric_limit_Lgamma = self.calorimetric_sfr_to_Lgamma(calorimetric_limit_sfr)
        ax.plot(
            calorimetric_limit_sfr.to(u.Msun/u.yr).value,
            calorimetric_limit_Lgamma.to(u.erg/u.s).value,
            color='black', linestyle=':', linewidth=1.5, alpha=0.8,
            label='Calorimetric limit (C19)'
        )

        # Configure the main axis
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        ax.set_xlim(self.x_range[0], self.x_range[-1])
        ax.legend()
    
    def calorimetric_function(self, SFR):
        """Calculate calorimetric gamma-ray luminosity for a given SFR"""
        # SFR is already in M_sun/yr units
        calorimetric_Lgamma = 6.7e39 * SFR  # erg/s
        return calorimetric_Lgamma
    
    def calorimetric_sfr_to_Lgamma(self, sfr):
        """Legacy method - use calorimetric_function instead"""
        if hasattr(sfr, 'to'):  # If it has units
            return 6.7e39 * sfr.to(u.Msun/u.yr).value * u.erg/u.s
        else:  # If it's just a number
            return 6.7e39 * sfr * u.erg/u.s

    def plot_residuals(self, ax):
        """Plot residuals (data/calorimetric limit) in the lower panel using a log scale"""
        # Track all ratios to set y-limits later
        all_ratios = []
        
        # Plot simulation residuals relative to the calorimetric limit if enabled
        if self.plot_sim_residuals:
            # Calculate calorimetric values for simulation data
            y_calorimetric = self.calorimetric_function(self.sim_table['SFR (M_sun/yr)'])
            ratios = self.sim_table['L_gamma (erg/s)'] / y_calorimetric
            all_ratios.extend(ratios)
            
            # Plot ratios for each point
            for i in range(self.sim_table.shape[0]):
                galaxy = self.sim_table['galaxy'][i]
                ax.scatter(
                    self.sim_table['SFR (M_sun/yr)'][i], ratios[i],
                    marker=self.get_marker(galaxy=galaxy), 
                    color=self.get_color(galaxy), 
                    zorder=3, s=60, alpha=0.7, 
                    edgecolor='None'
                )
        
        # Plot observation residuals relative to calorimetric limit if enabled
        if self.plot_obs_fit_residuals and self.obs_fit_result:
            x_data = self.obs_fit_result['data_x']
            y_data = self.obs_fit_result['data_y']
            # Calculate calorimetric values for observational data
            y_calorimetric = self.calorimetric_function(x_data)
            # Calculate ratios (data/calorimetric)
            ratios = y_data / y_calorimetric
            all_ratios.extend(ratios)
            
            # Get the included/excluded mask to maintain consistency with main plot
            if hasattr(self, 'galaxies_to_exclude') and hasattr(self, 'obs_data'):
                df_obs = self.obs_data['df']
                source_names = df_obs['Source'].values
                included_mask = ~np.isin(source_names, self.galaxies_to_exclude)
                
                # Plot included points with filled triangles
                ax.scatter(
                    x_data[included_mask], ratios[included_mask],
                    color='gray', marker='^', edgecolor='None', 
                    alpha=0.7, s=60, zorder=2
                )
                
                # Plot excluded points with empty triangles
                if np.any(~included_mask):
                    ax.scatter(
                        x_data[~included_mask], ratios[~included_mask],
                        facecolors='none', marker='^', 
                        edgecolor='gray', alpha=0.7, s=60, zorder=2
                    )
            else:
                # Fallback: plot all points with filled triangles
                ax.scatter(
                    x_data, ratios,
                    color='gray', marker='^', edgecolor='None', alpha=0.7, s=60, zorder=2
                )
        
        # Plot the observational fit relative to calorimetric limit
        if self.plot_obs_fit_main and hasattr(self, 'obs_fit_function'):
            x_range = self.x_range
            y_obs_fit = self.obs_fit_function(x_range)
            y_calorimetric_fit = self.calorimetric_function(x_range)
            obs_fit_ratio = y_obs_fit / y_calorimetric_fit
            
            ax.plot(x_range, obs_fit_ratio, '--', color='gray', 
                   linewidth=1.5, alpha=0.7, zorder=1,
                   label='Obs. fit / Calorimetric')

        # Add a horizontal line at y=1 (calorimetric limit)
        ax.axhline(y=1, color='black', linestyle=':', linewidth=1.5, 
                  zorder=1, alpha=0.8, label='Calorimetric limit (K98)')

        # Configure the axis
        ax.set_yscale('log')
        #if all_ratios:
        #    ratio = 10**(1.2*np.max(np.abs(np.log10(np.array(all_ratios)))))
        #    ax.set_ylim(1/ratio, ratio)
        ax.set_ylim(1e-2, 8)
        ax.set_ylabel(r'$L_{\gamma}/L_{\gamma,\mathrm{cal}}$')
    
    def finalize(self, fig, axes):
        """Finalize plot with labels, scales, and save"""
        # Configure upper panel (with data)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        axes[0].set_xlim(self.x_range[0], self.x_range[-1])
        axes[0].set_ylim(1e34, 1e43)
        axes[0].legend()
        
        # Configure lower panel (with residuals)
        axes[1].set_xscale('log')
        axes[1].set_xlabel(r'$\mathrm{SFR}~(M_\odot/\mathrm{yr})$')
        axes[1].set_xlim(self.x_range[0], self.x_range[-1])
        
        plt.tight_layout()
        print(f'Saving figure to {self.output_filename}')
        fig.savefig(self.output_filename, dpi=300)

def main(E_min=1*u.GeV, E_max=1000*u.GeV, show_names=False):
    print('Do not run this file directly; it is meant to be imported and used as a module.')
    '''
    Example usage:
    print('Running SFR_Lgamma.py')
    galaxies = {
        'm12f_cd': [600],
        'm12i_et': [60], 
        'm12i_sc_fx10': [60], 
        'm12i_sc_fx100': [60],
        'm12i_cd': [600],
        'm12r_cd': [600],
        'm12w_cd': [600],
        'm11b_cd': [600],
        #'m11b_cd_007': [600],
        #'m11b_cd_070': [600],
        #'m11b_cd_210': [600],
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
    '''

if __name__ == '__main__':
    main(E_min=1*u.GeV, E_max=1000*u.GeV, show_names=False)

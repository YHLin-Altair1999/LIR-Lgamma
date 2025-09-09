import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from My_Plugin.quantity import L_IR, L_gamma_yt, L_gamma_YHLin
from My_Plugin.LoadData import get_snap_path, get_center
import yt
import os
import pickle
from glob import glob
from astropy.cosmology import Planck18, z_at_value, FlatLambdaCDM
import astropy.units as u
import astropy.constants as c
from scipy import integrate
from scipy.optimize import curve_fit
# Import the plotting function from plot_obs
from plot_obs import plot_from_pickle
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
    })

class LIR_Lgamma_Plot:
    def __init__(self, 
        galaxies=None,
        E_min=1*u.GeV, E_max=1000*u.GeV, 
        Lgamma_profile_folder='/tscc/lustre/ddn/scratch/yel051/tables/Lgamma_profiles',
        sim_table_path='./tables/Lgamma_LIR.csv',
        sed_base_path=f'/tscc/lustre/ddn/scratch/yel051/SKIRT/output/',
        obs_pickle_path='./obs_data/fit_results.pkl',  # Parameter for pickle file
        show_obs_gal_name=False,
        show_sim_gal_name=False
        ):
        """Initialize LIR_Lgamma_Plot with parameters for analysis and visualization"""
        # Configuration parameters
        self.galaxies = galaxies
        self.E_min = E_min
        self.E_max = E_max
        self.Lgamma_profile_folder = Lgamma_profile_folder
        self.sim_table_path = sim_table_path
        self.sed_base_path = sed_base_path
        self.obs_pickle_path = obs_pickle_path  # Store pickle path
        self.x_range = np.logspace(6.5, 12.5, 100)
        self.cosmo = Planck18
        
        # All the controls related to what to plot
        self.show_obs_gal_name = show_obs_gal_name
        self.show_obs_errorbars = False
        self.show_sim_gal_name = show_sim_gal_name
        self.plot_obs_fit_main = True
        self.plot_obs_fit_residuals = True
        self.plot_sim_fit_main = False
        self.plot_sim_residuals = True

        # Data storage attributes
        self.sim_data = None  # Simulation dataframe
        self.obs_data = None  # Observational dataframe and calculated values
        self.sim_fit_result = None  # Simulation fit results
        self.obs_fit_result = None  # Observation fit results
        self.fig = None  # Figure for plotting
        self.axes = None  # Axes for plotting

        self.calorimetric_limit_FIR = self.x_range * u.Lsun  # Calorimetric FIR luminosity
        self.epsilon = 0.79 # correcting factor for IMF
        self.calorimetric_limit_sfr = self.calorimetric_limit_FIR/u.Lsun * self.epsilon * 1.7e-10  * u.M_sun / u.yr # SFR in M_sun/yr
        self.calorimetric_limit_Lgamma = 6.7e39 * self.calorimetric_limit_sfr.to(u.Msun/u.yr).value * u.erg/u.s


    def calculate_LIR(self, galaxy, snap):
        path = os.path.join(self.sed_base_path, f'{galaxy}/snap_{snap}/run_SKIRT_i00_sed.dat')
        LIR = L_IR(path)
        return LIR

    def calculate_Lgamma(self, galaxy, snap, mode='yt', aperture=25*u.kpc):
        """Calculate gamma ray luminosity using either yt or YHLin method"""
        aperture = aperture or self.aperture
        
        if mode == 'yt':
            print('Calculating gamma ray luminosity using yt')
            path = get_snap_path(galaxy, snap)
            ds = yt.load(glob(os.path.join(path, "*.hdf5"))[0])
            out = L_gamma_yt(ds, get_center(galaxy, snap))
        else:
            out = L_gamma_YHLin(galaxy, snap, aperture)
        return out

    def make_sim_table(self):
        """Create or update table with gamma ray and IR luminosities"""

        data = []
        
        for galaxy, snaps in self.galaxies.items():
            for snap in snaps:
                fname = os.path.join(self.Lgamma_profile_folder, f'Lgamma_profile_{galaxy}_snap{snap:03d}.npy')
                profile = np.load(fname)
                Lgamma = np.sum(profile[:,1])
                data.append({
                    'galaxy': galaxy, 
                    'snap': snap, 
                    'L_gamma (erg/s)': Lgamma,
                    #'L_gamma (erg/s)': self.calculate_Lgamma(galaxy, snap, mode='YHLin', aperture=aperture).to('erg/s').value, 
                    'L_IR (L_sun)': self.calculate_LIR(galaxy, snap).to('L_sun').value
                    })
        df = pd.DataFrame(data)
        df = df.sort_values(by='snap')
        df.to_csv(self.sim_table_path, index=False)
        self.sim_table = df

    def get_marker(self, galaxy: str) -> str:
        gal_type = ''.join(galaxy.split('_')[1:])
        match gal_type:
            case 'cd':
                marker = 'o'
            case _:
                marker = 's'
        return marker

    def get_color(self, galaxy: str) -> str:
        gal_type = galaxy.split('_')[1]
        match gal_type:
            case 'cd':
                color = 'C0'
            case 'et':
                color = 'C1'
            case 'sc':
                color = 'C2'
        return color

    def linear_fit(self, x, alpha, beta):
        return alpha * x + beta

    def _fit_powerlaw(self, x_data, y_data, label_prefix='', clip_limit=3, ax=None):
        """Helper method to perform power-law fitting with sigma clipping"""
        # Calculate reference point for normalization
        ref_point = np.median(x_data)
        log_x = np.log10(x_data/ref_point)  # Convert to log and normalize
        log_y = np.log10(y_data)  # Convert to log
        
        # Initial fit
        params_initial, pcov_initial = curve_fit(self.linear_fit, log_x, log_y)
        alpha_initial, beta_initial = params_initial
        alpha_initial_error, beta_initial_error = np.sqrt(np.diag(pcov_initial))
        print(f"{label_prefix} initial fit: Alpha: {alpha_initial:.2f} ± {alpha_initial_error:.2f}, Beta: {beta_initial:.2f} ± {beta_initial_error:.2f}")
        
        # Calculate residuals for sigma clipping
        y_pred_initial = self.linear_fit(log_x, alpha_initial, beta_initial)
        residuals = log_y - y_pred_initial
        
        # Sigma clipping
        sigma = np.std(residuals)
        mask = np.abs(residuals) < clip_limit*sigma
        
        # Re-fit with clipped data
        x_clipped = log_x[mask]
        y_clipped = log_y[mask]
        
        # Perform the curve fitting with clipped data
        params, pcov = curve_fit(self.linear_fit, x_clipped, y_clipped)
        alpha, beta = params
        perr = np.sqrt(np.diag(pcov))
        alpha_error, beta_error = perr
        print(f"{label_prefix} fit: Alpha: {alpha:.2f} ± {alpha_error:.2f}, Beta: {beta:.2f} ± {beta_error:.2f}")
        
        # Print info about clipping
        num_removed = len(log_x) - len(x_clipped)
        print(f"{label_prefix} fit: Removed {num_removed} outliers beyond {clip_limit}-sigma ({num_removed/len(log_x)*100:.1f}%)")
        
        # Return best-fit parameters and function for calculating residuals
        fit_function = lambda x: 10**(self.linear_fit(np.log10(x/ref_point), alpha, beta))
        
        return {
            'alpha': alpha,
            'alpha_error': alpha_error,
            'beta': beta,
            'beta_error': beta_error,
            'ref_point': ref_point,
            'fit_function': fit_function,
            'data_x': x_data,
            'data_y': y_data,
            'mask': mask
        }



    def load_obs_data(self):
        """Load observational data from pickle file generated by fit_obs_data.py"""
        # Load the pickle file with all the fit results
        with open(self.obs_pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        
        print(f"Loaded observational data from {self.obs_pickle_path}")
        
        # Extract the relevant data
        df = pickle_data['dataframe']
        L_gammas = df['L_gamma (erg/s)'].values
        
        # Calculate error bars from the asymmetric errors
        L_gammas_error = np.vstack([
            df['L_gamma_err_minus'].values,
            df['L_gamma_err_plus'].values
        ])
        
        # Store results in the expected format
        self.obs_data = {
            'df': df,
            'L_gammas': L_gammas,
            'L_gammas_error': L_gammas_error
        }
        
        # Store the galaxies to exclude from fitting (from pickle)
        self.galaxies_to_exclude = pickle_data['data']['excluded_galaxies']
        
        print(f"Loaded {len(L_gammas)} galaxies, excluding {len(self.galaxies_to_exclude)} from fits")



    def plot_obs_from_pickle(self, ax, pickle_file='fit_results.pkl', interval_type='prediction'):
        """Plot observational data using the method from plot_obs.py"""
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
        
        x_data = data['L_IR_Lsun']
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
        
        # Plot fit line and confidence interval
        x_fit = np.logspace(np.log10(x_data.min()) - 0.5, np.log10(x_data.max()) + 0.5, 1000)
        x_fit = self.x_range
        A = fit_params['A']
        B = fit_params['B']
        y_fit = A * x_fit**B
        
        # Calculate confidence/prediction interval using the saved parameters
        if 'confidence_function' in fit_result:
            y_lower, y_upper = fit_result['confidence_function'](x_fit, interval_type)
        elif 'confidence_params' in fit_result:
            # Recreate confidence intervals from stored parameters
            params = fit_result['confidence_params']
            log_x_plot = np.log10(x_fit)
            
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
        fit_label = r'Best fit: $L_\gamma = {:.2e} \cdot L_\mathrm{{IR}}^{{{:.2f}}}$'.format(A, B)
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
        
        # Add correlation coefficient
        r_value = fit_params['correlation_coefficient']
        p_value = fit_params['p_value']
        '''
        ax.text(0.05, 0.95, f'$r = {r_value:.3f}$\n$p = {p_value:.3e}$', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        '''
        # Store the fit result for residual plotting
        self.obs_fit_result = {
            'A': A,
            'B': B,
            'fit_function': lambda x: A * (x)**B,
            'data_x': x_data,
            'data_y': y_data
        }
        self.obs_fit_function = self.obs_fit_result['fit_function']
        
        print(f"Loaded observational data and fit from {pickle_file}")
        return True

    def fit_obs_data(self):
        """Load observational fit from pickle file"""
        # Load the pickle file with fit results
        with open(self.obs_pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        
        # Extract fit parameters from pickle
        fit_params = pickle_data['fit_parameters']
        fit_result_pickle = pickle_data['fit_result']
        
        # Convert to the format expected by the plotting functions
        ref_point = fit_result_pickle.get('ref_point', 1.0)
        A = fit_params['A']
        B = fit_params['B']
        A_err = fit_params['A_error']
        B_err = fit_params['B_error']
        
        # Create fit function: y = A * x^B
        def fit_function(x):
            return A * (x)**B
        
        # Convert to log-space parameters for compatibility with existing plotting code
        # The pickle stores: y = A * x^B
        # We need: log(y) = alpha * log(x/ref_point) + beta
        # So: log(A * x^B) = log(A) + B * log(x) = B * log(x) + log(A)
        # If we use ref_point = 1, then: alpha = B, beta = log10(A)
        alpha = B
        beta = np.log10(A)
        alpha_error = B_err
        beta_error = A_err / (A * np.log(10))  # Convert A_err to log space
        
        # Create the fit result dictionary
        self.obs_fit_result = {
            'alpha': alpha,
            'alpha_error': alpha_error,
            'beta': beta,
            'beta_error': beta_error,
            'ref_point': ref_point,
            'fit_function': fit_function,
            'data_x': pickle_data['data']['L_IR_Lsun'],
            'data_y': pickle_data['data']['L_gamma_erg_per_s']
        }
        
        # Store fit function for residuals
        self.obs_fit_function = fit_function
        
        print(f"Loaded fit from pickle: L_gamma = {A:.2e} * L_IR^{B:.3f}")
        print(f"Fit parameters: A = {A:.2e} ± {A_err:.2e}, B = {B:.3f} ± {B_err:.3f}")



    def calorimetric_function(self, L_IR):
        """Calculate calorimetric gamma-ray luminosity for a given IR luminosity"""
        # Convert L_IR to the calorimetric SFR and then to gamma-ray luminosity
        # L_IR is already in L_sun units, so we don't need unit conversion
        calorimetric_sfr = L_IR * self.epsilon * 1.7e-10  # SFR in M_sun/yr
        calorimetric_Lgamma = 6.7e39 * calorimetric_sfr  # erg/s
        return calorimetric_Lgamma

    def plot_residuals(self, ax):
        """Plot residuals (data/calorimetric limit) in the lower panel using a log scale"""
        # Track all ratios to set y-limits later
        all_ratios = []
        
        # Plot simulation residuals relative to the calorimetric limit if enabled
        if self.plot_sim_residuals:
            # Calculate calorimetric values for simulation data
            y_calorimetric = self.calorimetric_function(self.sim_table['L_IR (L_sun)'])
            ratios = self.sim_table['L_gamma (erg/s)'] / y_calorimetric
            all_ratios.extend(ratios)
            
            # Plot ratios for each point
            for i in range(self.sim_table.shape[0]):
                galaxy = self.sim_table['galaxy'][i]
                ax.scatter(
                    self.sim_table['L_IR (L_sun)'][i], ratios[i],
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
            # We need to recreate this from the galaxies_to_exclude list
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
        ax.set_ylim(0.01, 8)
        ax.set_ylabel(r'$L_{\gamma}/L_{\gamma,\mathrm{calorimetric}}$')

    def finalize(self, fig, axes):
        """Finalize plot with labels, scales, and save"""
        
        # Configure upper panel (with data)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        axes[0].set_xlim(self.x_range[0], self.x_range[-1])
        #axes[0].set_ylim(1e36, 1e43)
        axes[0].set_ylim(1e34, 1e43)
        axes[0].legend()
        
        # Configure lower panel (with residuals)
        axes[1].set_xscale('log')
        axes[1].set_xlabel(r'$L_{\rm IR, ~8-1000 ~\mu m} ~(L_\odot)$')
        axes[1].set_xlim(self.x_range[0], self.x_range[-1])
        
        plt.tight_layout()
        fig.savefig('LIR-Lgamma.png', dpi=300)

    def run(self):
        """Run the full analysis and plotting pipeline"""
        # Create figure with two panels (or one if residuals not needed)
        if self.plot_obs_fit_residuals or self.plot_sim_residuals:
            self.fig, self.axes = plt.subplots(
                figsize=(4,4), 
                nrows=2, ncols=1, 
                sharex=True,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
            )
        else:
            self.fig, ax = plt.subplots(figsize=(4,4.5))
            self.axes = [ax, None]

        # Process data
        self.make_sim_table()
        self.load_obs_data()
        
        # Perform fitting
        self.fit_obs_data()
        
        # Create plots for the main panel (combines plot_sim and plot_obs)
        sim_result = self.plot_main(ax=self.axes[0])
        
        
        # Handle residuals if needed
        if self.plot_obs_fit_residuals or self.plot_sim_residuals:
            # Hide x tick labels for top panel
            plt.setp(self.axes[0].get_xticklabels(), visible=False)
            
            # Plot residuals
            self.plot_residuals(self.axes[1])
        
        # Finalize the plot
        self.finalize(self.fig, self.axes)
        
        return self.fig, self.axes

    def _plot_fit_result(self, ax, fit_result, color='blue', label=None):
        """Plot fit result line and confidence band on the given axis
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis to plot on
        fit_result : dict
            The dictionary containing fit parameters and function
        color : str
            Color to use for the fit line and confidence band
        label : str, optional
            Label for the fit line in the legend
        """
        if fit_result is None:
            return
            
        # Extract parameters
        alpha = fit_result['alpha']
        alpha_error = fit_result['alpha_error']
        beta = fit_result['beta']
        beta_error = fit_result['beta_error']
        ref_point = fit_result['ref_point']
        
        # Calculate best fit line over the full range
        x_range = self.x_range
        log_x_range = np.log10(x_range/ref_point)
        
        # Calculate best fit line
        log_y_bestfit = self.linear_fit(log_x_range, alpha, beta)
        y_bestfit = 10**log_y_bestfit
        
        # Plot the best fit line
        line_label = label or f"Best fit: $L_\\gamma \\propto L_\\mathrm{{IR}}^{{{alpha:.2f} \\pm {alpha_error:.2f}}}$"
        ax.plot(x_range, y_bestfit, color=color, linestyle='--', zorder=1, label=line_label, alpha=0.1)
        
        # Calculate uncertainty band points
        log_y_1 = self.linear_fit(log_x_range, alpha+alpha_error, beta+beta_error)
        log_y_2 = self.linear_fit(log_x_range, alpha-alpha_error, beta-beta_error)
        log_y_3 = self.linear_fit(log_x_range, alpha+alpha_error, beta-beta_error)
        log_y_4 = self.linear_fit(log_x_range, alpha-alpha_error, beta+beta_error)
        
        # Convert back to linear space
        y_1 = 10**log_y_1
        y_2 = 10**log_y_2
        y_3 = 10**log_y_3
        y_4 = 10**log_y_4
        
        # Plot the confidence band
        stacked = np.vstack((y_1, y_2, y_3, y_4))
        ax.fill_between(x_range, np.min(stacked, axis=0), np.max(stacked, axis=0),
            color=color, alpha=0.1, zorder=1)

    def plot_main(self, ax=None, sim_table_path=None):
        """Plot both simulation and observational data on the main panel"""
        sim_table_path = sim_table_path or self.sim_table_path
        ax = ax or self.axes[0]
        
        # Plot simulation data
        df = pd.read_csv(sim_table_path)
        for i in range(df.shape[0]):
            galaxy = df['galaxy'][i]
            ax.scatter(
                df['L_IR (L_sun)'][i], df['L_gamma (erg/s)'][i], 
                marker=self.get_marker(galaxy=galaxy), color=self.get_color(galaxy), 
                zorder=3, s=60, alpha=0.7, edgecolor='None')
            # Add label with galaxy name using text
            if self.show_sim_gal_name:
                ax.text(
                    df['L_IR (L_sun)'][i] * 1.05,  # Slight offset to the right
                    df['L_gamma (erg/s)'][i], 
                    galaxy,
                    color='C0',
                    fontsize=12,
                    alpha=1
                )
        
        
        # Plot simulation fit if enabled
        if self.plot_sim_fit_main:
            # Fit power law for simulation data
            self.sim_fit_result = self._fit_powerlaw(
                x_data=df['L_IR (L_sun)'],
                y_data=df['L_gamma (erg/s)'],
                label_prefix='Sim'
            )
            self._plot_fit_result(ax, self.sim_fit_result, color='skyblue', label='Simulation fit')
        
        # Plot observational data using pickle file method
        self.plot_obs_from_pickle(ax, self.obs_pickle_path, interval_type='prediction')

        ax.plot(
            self.calorimetric_limit_FIR.to('L_sun').value,
            self.calorimetric_limit_Lgamma.to('erg/s').value,
            color='black', linestyle=':', linewidth=1.5, alpha=0.8,
            label='Calorimetric limit (K98)'
        )

        # Configure the main axis
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        ax.set_xlim(self.x_range[0], self.x_range[-1])
        ax.legend()
        
        return


def main():
    '''
    Usage example for LIR_Lgamma_Plot class
    '''
    print('Running LIR_Lgamma.py')
    galaxies = {
        'm12i_cd': [600],
    }
    
    # Create the plotter with our parameters
    plotter = LIR_Lgamma_Plot(
        galaxies=galaxies,
        E_min=1*u.GeV, 
        E_max=1000*u.GeV,
        show_obs_gal_name=True
    )
    
    plotter.run()

if __name__ == '__main__':
    main()

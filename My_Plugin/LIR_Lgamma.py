import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from My_Plugin.quantity import L_IR, L_gamma_yt, L_gamma_YHLin
from My_Plugin.LoadData import get_snap_path, get_center
import yt
import os
import pickle
from glob import glob
import astropy.units as u
from scipy.optimize import curve_fit
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
        sed_base_path=f'/tscc/lustre/ddn/scratch/yel051/SKIRT/output/Fiducial/',
        obs_pickle_path='./obs_data/fit_results.pkl',
        show_obs_gal_name=False,
        show_sim_gal_name=False,
        show_calorimetric_limit=True,
        output_filename='LIR_Lgamma.png',
        figsize=(4, 4),
        x_range=np.logspace(6.5, 12.5, 100)
        ):
        """Initialize LIR_Lgamma_Plot with parameters for analysis and visualization"""
        self.galaxies = galaxies
        self.E_min = E_min
        self.E_max = E_max
        self.Lgamma_profile_folder = Lgamma_profile_folder
        self.sim_table_path = sim_table_path
        self.sed_base_path = sed_base_path
        self.obs_pickle_path = obs_pickle_path
        self.x_range = x_range
        
        # Plot controls
        self.show_obs_gal_name = show_obs_gal_name
        self.show_obs_errorbars = False
        self.show_sim_gal_name = show_sim_gal_name
        self.show_calorimetric_limit = show_calorimetric_limit
        self.plot_obs_fit_main = True
        self.plot_obs_fit_residuals = True
        self.plot_sim_fit_main = False
        self.plot_sim_residuals = True
        self.output_filename = output_filename
        self.figsize = figsize

        # Data storage
        self.sim_data = None
        self.obs_data = None
        self.sim_fit_result = None
        self.obs_fit_result = None
        self.fig = None
        self.axes = None

        # Calorimetric parameters
        self.epsilon = 0.79


    def calculate_LIR(self, galaxy, snap):
        path = os.path.join(self.sed_base_path, f'{galaxy}/snap_{snap}/run_SKIRT_i00_sed.dat')
        return L_IR(path)

    def calculate_Lgamma(self, galaxy, snap, mode='yt', aperture=25*u.kpc):
        """Calculate gamma ray luminosity using either yt or YHLin method"""
        if mode == 'yt':
            path = get_snap_path(galaxy, snap)
            ds = yt.load(glob(os.path.join(path, "*.hdf5"))[0])
            return L_gamma_yt(ds, get_center(galaxy, snap))
        return L_gamma_YHLin(galaxy, snap, aperture)

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
            case _:
                color = 'black'
        return color

    def linear_fit(self, x, alpha, beta):
        return alpha * x + beta

    def _fit_powerlaw(self, x_data, y_data, label_prefix=''):
        """Simple power-law fitting without sigma clipping"""
        log_x = np.log10(x_data)
        log_y = np.log10(y_data)
        
        params, pcov = curve_fit(self.linear_fit, log_x, log_y)
        alpha, beta = params
        alpha_error, beta_error = np.sqrt(np.diag(pcov))
        
        fit_function = lambda x: 10**(self.linear_fit(np.log10(x), alpha, beta))
        
        return {
            'alpha': alpha,
            'alpha_error': alpha_error,
            'beta': beta,
            'beta_error': beta_error,
            'ref_point': 1.0,
            'fit_function': fit_function,
            'data_x': x_data,
            'data_y': y_data
        }

    def load_obs_data(self):
        """Load observational data from pickle file generated by fit_obs_data.py"""
        with open(self.obs_pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        
        df = pickle_data['dataframe']
        L_gammas = df['L_gamma (erg/s)'].values
        L_gammas_error = np.vstack([
            df['L_gamma_err_minus'].values,
            df['L_gamma_err_plus'].values
        ])
        
        self.obs_data = {
            'df': df,
            'L_gammas': L_gammas,
            'L_gammas_error': L_gammas_error
        }
        self.galaxies_to_exclude = pickle_data['data']['excluded_galaxies']

    def _plot_obs_data_points(self, ax, x_data, y_data, yerr, included_mask):
        """Plot observational data points with consistent styling"""
        ax.scatter(
            x_data[included_mask], y_data[included_mask],
            marker='^', color='gray', edgecolor='None',
            s=49, alpha=0.7, zorder=3,
            label='Ambrosone et al. (2024)'
        )
        
        if np.any(~included_mask):
            ax.scatter(
                x_data[~included_mask], y_data[~included_mask],
                facecolors='none', marker='^', 
                edgecolor='gray', alpha=0.7, s=49, zorder=2
            )
        
        if self.show_obs_errorbars:
            ax.errorbar(
                x_data, y_data, yerr=yerr,
                fmt='None', ecolor='gray', capsize=3, alpha=0.7, zorder=2
            )

    def _plot_obs_fit_line(self, ax, x_range, A, B, fit_result, interval_type='prediction'):
        """Plot observational fit line and confidence intervals"""
        y_fit = A * x_range**B
        ax.plot(x_range, y_fit, '--', color='gray', zorder=4, alpha=0.5)
        
        # Recreate confidence intervals from stored parameters
        params = fit_result['confidence_params']
        log_x_plot = np.log10(x_range)
        
        if interval_type == 'confidence':
            se = params['residual_std'] * np.sqrt(1/params['n_points'] + (log_x_plot - params['x_mean'])**2 / params['sum_x_sq'])
        else:  # prediction interval
            se = params['residual_std'] * np.sqrt(1 + 1/params['n_points'] + (log_x_plot - params['x_mean'])**2 / params['sum_x_sq'])
        
        log_y_plot = params['actual_intercept'] + params['slope'] * log_x_plot
        y_upper = 10**(log_y_plot + se)
        y_lower = 10**(log_y_plot - se)
        
        # Plot confidence/prediction interval
        ax.fill_between(x_range, y_lower, y_upper, color='gray', alpha=0.2, zorder=2)
        
        return y_fit

    def _add_galaxy_annotations(self, ax, df, x_data, y_data):
        """Add galaxy name annotations if enabled"""
        if self.show_obs_gal_name:
            for i in range(len(y_data)):
                ax.annotate(
                    df['Source'].iloc[i], 
                    (x_data[i], y_data[i]),
                    xytext=(5, 0), 
                    textcoords='offset points',
                    fontsize=8, alpha=0.7
                )

    def plot_obs_from_pickle(self, ax, pickle_file='fit_results.pkl', interval_type='prediction'):
        """Plot observational data using the method from plot_obs.py"""
        with open(pickle_file, 'rb') as f:
            results = pickle.load(f)
        
        fit_params = results['fit_parameters']
        data = results['data']
        df = results['dataframe']
        fit_result = results['fit_result']  # Get fit_result for confidence intervals
        
        excluded_galaxies = data['excluded_galaxies']
        source_names = data['source_names']
        included_mask = ~np.isin(source_names, excluded_galaxies)
        
        x_data = data['L_IR_Lsun']
        y_data = data['L_gamma_erg_per_s']
        yerr = [df['L_gamma_err_minus'].values, df['L_gamma_err_plus'].values]
        
        # Plot data points
        self._plot_obs_data_points(ax, x_data, y_data, yerr, included_mask)
        
        # Plot fit line with confidence intervals
        A, B = fit_params['A'], fit_params['B']
        self._plot_obs_fit_line(ax, self.x_range, A, B, fit_result, interval_type)
        
        # Add galaxy annotations
        self._add_galaxy_annotations(ax, df, x_data, y_data)
        
        # Store fit result for residuals
        self.obs_fit_result = {
            'A': A, 'B': B,
            'fit_function': lambda x: A * (x)**B,
            'data_x': x_data, 'data_y': y_data
        }
        self.obs_fit_function = self.obs_fit_result['fit_function']
        return True

    def fit_obs_data(self):
        """Load observational fit from pickle file"""
        with open(self.obs_pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        
        fit_params = pickle_data['fit_parameters']
        A = fit_params['A']
        B = fit_params['B']
        
        fit_function = lambda x: A * (x)**B
        
        # Convert to log-space parameters for compatibility
        alpha = B
        beta = np.log10(A)
        alpha_error = fit_params['B_error']
        beta_error = fit_params['A_error'] / (A * np.log(10))
        
        self.obs_fit_result = {
            'alpha': alpha,
            'alpha_error': alpha_error,
            'beta': beta,
            'beta_error': beta_error,
            'ref_point': 1.0,
            'fit_function': fit_function,
            'data_x': pickle_data['data']['L_IR_Lsun'],
            'data_y': pickle_data['data']['L_gamma_erg_per_s']
        }
        self.obs_fit_function = fit_function



    def calorimetric_function(self, L_IR):
        """Calculate calorimetric gamma-ray luminosity for a given IR luminosity"""
        # Convert L_IR to the calorimetric SFR and then to gamma-ray luminosity
        # L_IR is already in L_sun units, so we don't need unit conversion
        calorimetric_sfr = L_IR * self.epsilon * 1.7e-10  # SFR in M_sun/yr
        calorimetric_Lgamma = 6.7e39 * calorimetric_sfr  # erg/s
        return calorimetric_Lgamma

    def plot_residuals(self, ax):
        """Plot residuals (data/calorimetric limit) in the lower panel"""
        # Plot simulation residuals if enabled
        if self.plot_sim_residuals and hasattr(self, 'sim_table'):
            y_calorimetric = self.calorimetric_function(self.sim_table['L_IR (L_sun)'])
            ratios = self.sim_table['L_gamma (erg/s)'] / y_calorimetric
            
            for i in range(self.sim_table.shape[0]):
                galaxy = self.sim_table['galaxy'][i]
                ax.scatter(
                    self.sim_table['L_IR (L_sun)'][i], ratios[i],
                    marker=self.get_marker(galaxy=galaxy), 
                    color=self.get_color(galaxy), 
                    zorder=3, s=60, alpha=0.7, edgecolor='None'
                )
        
        # Plot observation residuals if enabled
        if self.plot_obs_fit_residuals and hasattr(self, 'obs_fit_result'):
            # Load the mask information from pickle to determine included/excluded galaxies
            with open(self.obs_pickle_path, 'rb') as f:
                results = pickle.load(f)
            
            data = results['data']
            excluded_galaxies = data['excluded_galaxies']
            source_names = data['source_names']
            included_mask = ~np.isin(source_names, excluded_galaxies)
            
            x_data = self.obs_fit_result['data_x']
            y_data = self.obs_fit_result['data_y']
            y_calorimetric = self.calorimetric_function(x_data)
            ratios = y_data / y_calorimetric
            
            # Plot included galaxies (filled points)
            ax.scatter(
                x_data[included_mask], ratios[included_mask], 
                color='gray', marker='^', 
                edgecolor='None', alpha=0.7, s=60, zorder=2
            )
            
            # Plot excluded galaxies (hollow points)
            if np.any(~included_mask):
                ax.scatter(
                    x_data[~included_mask], ratios[~included_mask], 
                    facecolors='none', marker='^', 
                    edgecolor='gray', alpha=0.7, s=60, zorder=2
                )
        
        # Plot observational fit relative to calorimetric limit
        if self.plot_obs_fit_main and hasattr(self, 'obs_fit_function'):
            y_obs_fit = self.obs_fit_function(self.x_range)
            y_calorimetric_fit = self.calorimetric_function(self.x_range)
            obs_fit_ratio = y_obs_fit / y_calorimetric_fit
            
            ax.plot(self.x_range, obs_fit_ratio, '--', color='gray', 
                   linewidth=1.5, alpha=0.7, zorder=1,
                   label='Obs. fit / Calorimetric')

        # Add calorimetric limit reference line
        ax.axhline(y=1, color='black', linestyle=':', linewidth=1.5, 
                  zorder=1, alpha=0.8, label='Calorimetric limit (K98)')

        ax.set_yscale('log')
        ax.set_ylim(0.01, 8)
        ax.set_ylabel(r'$L_{\gamma}/L_{\gamma,\mathrm{cal}}$')

    def finalize(self, fig, axes):
        """Finalize plot with labels, scales, and save"""
        
        # Configure upper panel (with data)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        axes[0].set_xlim(self.x_range[0], self.x_range[-1])
        #axes[0].set_ylim(1e36, 1e43)
        #axes[0].set_ylim(1e34, 1e43)
        axes[0].legend()
        
        # Configure lower panel (with residuals)
        axes[1].set_xscale('log')
        axes[1].set_xlabel(r'$L_{\rm IR, ~8-1000 ~\mu m} ~(L_\odot)$')
        axes[1].set_xlim(self.x_range[0], self.x_range[-1])
        
        plt.tight_layout()
        print(f'Saving figure to {self.output_filename}')
        fig.savefig(self.output_filename, dpi=300)

    def run(self):
        """Run the full analysis and plotting pipeline"""
        # Create figure with two panels (or one if residuals not needed)
        if self.plot_obs_fit_residuals or self.plot_sim_residuals:
            self.fig, self.axes = plt.subplots(
                figsize=self.figsize, 
                nrows=2, ncols=1, 
                sharex=True,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
            )
        else:
            self.fig, ax = plt.subplots(figsize=self.figsize)
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
            
            if self.show_sim_gal_name:
                ax.text(
                    df['L_IR (L_sun)'][i] * 1.05,
                    df['L_gamma (erg/s)'][i], 
                    galaxy, color='C0', fontsize=12, alpha=1
                )
        
        # Plot simulation fit if enabled
        if self.plot_sim_fit_main:
            self.sim_fit_result = self._fit_powerlaw(
                x_data=df['L_IR (L_sun)'], y_data=df['L_gamma (erg/s)']
            )

        # Plot observational data
        self.plot_obs_from_pickle(ax, self.obs_pickle_path, interval_type='prediction')

        # Plot calorimetric limit if enabled
        if self.show_calorimetric_limit:
            calorimetric_FIR = self.x_range * u.Lsun
            calorimetric_sfr = calorimetric_FIR/u.Lsun * self.epsilon * 1.7e-10 * u.M_sun / u.yr
            calorimetric_Lgamma = 6.7e39 * calorimetric_sfr.to(u.Msun/u.yr).value * u.erg/u.s
            
            ax.plot(
                calorimetric_FIR.to('L_sun').value,
                calorimetric_Lgamma.to('erg/s').value,
                color='black', linestyle=':', linewidth=1.5, alpha=0.8,
                label='Calorimetric limit (K98)'
            )
        
        # Configure the axis
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        ax.set_xlim(self.x_range[0], self.x_range[-1])
        ax.legend()


def main():
    '''
    Usage example for LIR_Lgamma_Plot class
    '''
    print('Testing LIR_Lgamma.py')
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

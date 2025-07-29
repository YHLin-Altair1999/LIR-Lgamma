import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from My_Plugin.quantity import L_IR, L_gamma_yt, L_gamma_YHLin
from My_Plugin.LoadData import get_snap_path, get_center
import yt
import os
from glob import glob
from astropy.cosmology import Planck18, z_at_value, FlatLambdaCDM
import astropy.units as u
import astropy.constants as c
from scipy import integrate
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
        show_obs_gal_name=False
        ):
        """Initialize LIR_Lgamma_Plot with parameters for analysis and visualization"""
        # Configuration parameters
        self.galaxies = galaxies
        self.E_min = E_min
        self.E_max = E_max
        self.Lgamma_profile_folder = Lgamma_profile_folder
        self.sim_table_path = sim_table_path
        self.x_range = np.logspace(6.5, 12.5, 100)
        self.cosmo = Planck18
        
        # All the controls related to what to plot
        self.show_obs_gal_name = show_obs_gal_name
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
        

    def calculate_LIR(self, galaxy, snap):
        sed_path=f'/tscc/lustre/ddn/scratch/yel051/SKIRT/output/{galaxy}/snap_{snap}/run_SKIRT_i00_sed.dat'
        LIR = L_IR(sed_path)
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

    def plot_sim(self, sim_table_path=None, ax=None):
        """Plot simulation data on the given axis"""
        sim_table_path = sim_table_path or self.sim_table_path
        if ax is None:
            _, ax = plt.subplots(figsize=(5,4))
            
        df = pd.read_csv(sim_table_path)
        for i in range(df.shape[0]):
            galaxy = df['galaxy'][i]
            ax.scatter(
                df['L_IR (L_sun)'][i], df['L_gamma (erg/s)'][i], 
                marker=self.get_marker(galaxy=galaxy), color=self.get_color(galaxy), 
                zorder=3, s=60, alpha=0.7, edgecolor='None')
        
        # Fit power law and plot
        fit_result = self._fit_powerlaw(
            x_data=df['L_IR (L_sun)'],
            y_data=df['L_gamma (erg/s)'],
            label_prefix='Sim'
        )
        
        # Plot fit if enabled by control flag
        if self.plot_sim_fit_main:
            self._plot_fit_result(ax, fit_result, color='skyblue', label='Simulation fit')
        
        return ax, fit_result

    def luminosity(self, E, phi_0, gamma, D_L):
        z = z_at_value(self.cosmo.luminosity_distance, D_L)
        dFdE = E * phi_0*(E/u.GeV)**(-gamma)
        flux = integrate.simpson(dFdE.value, x=E.value) * (E.unit * dFdE.unit)
        L = 4*np.pi * D_L**2 * (1+z)**(gamma-2) * flux
        return L.to('erg/s').value

    def load_obs_data(self):
        """Load observational data"""
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
            x_data=df_for_fitting['L_IR (L_sun)'],
            y_data=df_for_fitting['L_gamma'],
            label_prefix='Obs'
        )
        
        # Store x and y data for residual calculation
        self.obs_fit_result['data_x'] = df['L_IR (L_sun)']
        self.obs_fit_result['data_y'] = L_gammas
        self.obs_fit_function = self.obs_fit_result['fit_function']

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

            x_data = self.sim_table['L_IR (L_sun)']
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
        #axes[0].set_ylim(1e36, 1e43)
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
                figsize=(5,5), 
                nrows=2, ncols=1, 
                sharex=True,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
            )
        else:
            self.fig, ax = plt.subplots(figsize=(5,4))
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
        
        # Fit power law for simulation data
        self.sim_fit_result = self._fit_powerlaw(
            x_data=df['L_IR (L_sun)'],
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
                df_obs['L_IR (L_sun)'], L_gammas,
                label='Ambrosone et al. (2024)', color='gray', marker='^', 
                edgecolor='None', alpha=0.7, s=60, zorder=2
            )
            
            # Add galaxy names next to data points if requested
            if self.show_obs_gal_name:
                for i in range(len(L_gammas)):
                    ax.annotate(
                        df_obs['Source'][i], 
                        (df_obs['L_IR (L_sun)'][i], L_gammas[i]),
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7
                    )
            
            # Plot the observation fit line and confidence band if enabled
            if self.plot_obs_fit_main and self.obs_fit_result:
                self._plot_fit_result(ax, self.obs_fit_result, color='gray', 
                                     label='Observation fit')
        
        calorimetric_limit_FIR = self.x_range * u.Lsun  # Calorimetric FIR luminosity
        epsilon = 0.79
        calorimetric_limit_sfr = calorimetric_limit_FIR/u.Lsun * epsilon * 1.7e-10  * u.M_sun / u.yr # SFR in M_sun/yr
        calorimetric_limit_Lgamma = 6.7e39 * calorimetric_limit_sfr.to(u.Msun/u.yr).value * u.erg/u.s

        ax.plot(
            calorimetric_limit_FIR.to('L_sun').value,
            calorimetric_limit_Lgamma.to('erg/s').value,
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
    print('Running LIR_Lgamma.py')
    galaxies = {
        #'m12f_cd': [600], 
        #'m12i_et': [60], 
        #'m12i_sc_fx10': [60], 
        #'m12i_sc_fx100': [60],
        #'m12i_cd': [600],
        'm12i_cd': np.arange(100,600,50),
        #'m12r_cd': [600],
        #'m12w_cd': [600],
        #'m11b_cd': [600],
        #'m11c_cd': [600],
        #'m11d_cd': [600],
        #'m11f_cd': [600],
        #'m11g_cd': [600],
        #'m11h_cd': [600],
        #'m11v_cd': [600],
        #'m10v_cd': [600],
        #'m09_cd': [600],
        #'m11f_et_AlfvenMax': [600],
        #'m11f_et_FastMax': [600],
        #'m11f_sc_fcas50': [600],
    }
    
    # Create the plotter with our parameters
    plotter = LIR_Lgamma_Plot(
        galaxies=galaxies,
        E_min=1*u.GeV, 
        E_max=1000*u.GeV,
        show_obs_gal_name=False
    )
    
    # Either run the full pipeline
    plotter.run()

if __name__ == '__main__':
    main()

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
        E_min=1*u.GeV, E_max=1000*u.GeV, # gamma ray luminosity energy band
        Lgamma_profile_folder='/tscc/lustre/ddn/scratch/yel051/tables/Lgamma_profiles',
        table_path='./tables/Lgamma_LIR.csv',
        aperture=25*u.kpc
        ):
        """Initialize LIR_Lgamma_Plot with parameters for analysis and visualization"""
        self.galaxies = galaxies
        self.E_min = E_min
        self.E_max = E_max
        self.Lgamma_profile_folder = Lgamma_profile_folder
        self.table_path = table_path
        self.aperture = aperture
        self.x_range = np.logspace(6, 12.5, 100)
        #self.cosmo = Planck18
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    def calculate_LIR(self, galaxy, snap):
        sed_path=f'/tscc/lustre/ddn/scratch/yel051/SKIRT/output/{galaxy}/snap_{snap}/run_SKIRT_i00_sed.dat'
        LIR = L_IR(sed_path)
        return LIR

    def calculate_Lgamma(self, galaxy, snap, mode='yt', aperture=None):
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

    def make_table(self, table_path=None, aperture=None):
        """Create or update table with gamma ray and IR luminosities"""
        table_path = table_path or self.table_path
        aperture = aperture or self.aperture
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
        df.to_csv(table_path, index=False)

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

    def _fit_powerlaw(self, x_data, y_data, color='blue', label_prefix='', clip_limit=3, ax=None):
        """Helper method to perform power-law fitting with sigma clipping
        
        Parameters:
        -----------
        x_data : array-like
            X values (typically LIR) for fitting
        y_data : array-like
            Y values (typically Lgamma) for fitting
        color : str
            Color for the fit line and uncertainty band
        label_prefix : str
            Prefix for the fit label (e.g. 'Simulation' or 'Observation')
        clip_limit : float
            Sigma limit for outlier rejection
        ax : matplotlib.axes, optional
            Axis to plot on
            
        Returns:
        --------
        dict
            Dictionary containing fit parameters
        """
        # Calculate reference point for normalization
        ref_point = np.median(x_data)
        log_x = np.log10(x_data/ref_point)  # Convert to log and normalize
        log_y = np.log10(y_data)  # Convert to log
        
        # Define linear function for fitting in log space
        def linear_fit(x, alpha, beta):
            return alpha * x + beta
        
        # Initial fit
        params_initial, pcov_initial = curve_fit(linear_fit, log_x, log_y)
        alpha_initial, beta_initial = params_initial
        alpha_initial_error, beta_initial_error = np.sqrt(np.diag(pcov_initial))
        print(f"{label_prefix} initial fit: Alpha: {alpha_initial:.2f} ± {alpha_initial_error:.2f}, Beta: {beta_initial:.2f} ± {beta_initial_error:.2f}")
        
        # Calculate residuals for sigma clipping
        y_pred_initial = linear_fit(log_x, alpha_initial, beta_initial)
        residuals = log_y - y_pred_initial
        
        # Sigma clipping
        sigma = np.std(residuals)
        mask = np.abs(residuals) < clip_limit*sigma
        
        # Re-fit with clipped data
        x_clipped = log_x[mask]
        y_clipped = log_y[mask]
        
        # Perform the curve fitting with clipped data
        params, pcov = curve_fit(linear_fit, x_clipped, y_clipped)
        alpha, beta = params
        perr = np.sqrt(np.diag(pcov))
        alpha_error, beta_error = perr
        print(f"{label_prefix} fit: Alpha: {alpha:.2f} ± {alpha_error:.2f}, Beta: {beta:.2f} ± {beta_error:.2f}")
        
        # Print info about clipping
        num_removed = len(log_x) - len(x_clipped)
        print(f"{label_prefix} fit: Removed {num_removed} outliers beyond {clip_limit}-sigma ({num_removed/len(log_x)*100:.1f}%)")
        
        if ax is not None:
            # Plot the best fit line
            x_range = self.x_range
            log_x_range = np.log10(x_range/ref_point)
            log_y_bestfit = linear_fit(log_x_range, alpha, beta)
            y_bestfit = 10**log_y_bestfit
            
            ax.plot(x_range, y_bestfit, color=color, linestyle='--', zorder=1)
            
            # Calculate and plot the uncertainty band
            log_y_1 = linear_fit(log_x_range, alpha+alpha_error, beta+beta_error)
            log_y_2 = linear_fit(log_x_range, alpha-alpha_error, beta-beta_error)
            log_y_3 = linear_fit(log_x_range, alpha+alpha_error, beta-beta_error)
            log_y_4 = linear_fit(log_x_range, alpha-alpha_error, beta+beta_error)
            
            # Convert back to linear space
            y_1 = 10**log_y_1
            y_2 = 10**log_y_2
            y_3 = 10**log_y_3
            y_4 = 10**log_y_4
            
            stacked = np.vstack((y_1, y_2, y_3, y_4))
            ax.fill_between(x_range, np.min(stacked, axis=0), np.max(stacked, axis=0),
                color=color, alpha=0.5, zorder=1)
        
        return {
            'alpha': alpha,
            'alpha_error': alpha_error,
            'beta': beta,
            'beta_error': beta_error,
            'ref_point': ref_point
        }

    def plot_sim(self, table_path=None, ax=None, plot_fit=False):
        """Plot simulation data on the given axis"""
        table_path = table_path or self.table_path
        if ax is None:
            _, ax = plt.subplots(figsize=(5,4))
            
        df = pd.read_csv(table_path)
        for i in range(df.shape[0]):
            galaxy = df['galaxy'][i]
            ax.scatter(
                df['L_IR (L_sun)'][i], df['L_gamma (erg/s)'][i], 
                marker=self.get_marker(galaxy=galaxy), color=self.get_color(galaxy), 
                zorder=3, s=60, alpha=0.8, edgecolor='None')
        
        # Fit power law and plot
        if plot_fit:
            self._fit_powerlaw(
                x_data=df['L_IR (L_sun)'],
                y_data=df['L_gamma (erg/s)'],
                color='blue',
                label_prefix='Sim',
                ax=ax
            )
        
        return ax

    def luminosity(self, E, phi_0, gamma, D_L):
            z = z_at_value(self.cosmo.luminosity_distance, D_L)
            dFdE = E * phi_0*(E/u.GeV)**(-gamma)
            flux = integrate.simpson(dFdE.value, x=E.value) * (E.unit * dFdE.unit)
            L = 4*np.pi * D_L**2 * (1+z)**(gamma-2) * flux
            return L.to('erg/s').value

    def plot_obs(self, ax=None, E_min=None, E_max=None, show_names=False):
        """Plot observational data on the given axis"""
        if ax is None:
            _, ax = plt.subplots(figsize=(5,4))
        E_min = E_min or self.E_min
        E_max = E_max or self.E_max
        
        obs_path = './obs_data/Ambrosone_2024.csv'
        df = pd.read_csv(obs_path)
        df['L_IR (L_sun)'] = np.array(df['LIR'])*1e10
        df['z'] = z_at_value(self.cosmo.luminosity_distance, np.array(df['DL'])*u.Mpc)
        L_gammas = np.array([
            self.luminosity(
                np.logspace(np.log10(E_min.to(u.GeV).value),np.log10(E_max.to(u.GeV).value),100) * u.GeV, 
                df['Phi0'][i] * 1e-12 * u.cm**-2 * u.s**-1 * u.MeV**-1, 
                df['gamma'][i], 
                df['DL'][i]*u.Mpc
                ) for i in range(len(df['z']))
            ])
        
        ax.scatter(
            df['L_IR (L_sun)'], L_gammas,
            label='Ambrosone et al. (2024)', color='gray', marker='^', edgecolor='None', alpha=0.8, s=60, zorder=2
            )
        
        # Add galaxy names next to data points
        if show_names:
            for i in range(len(L_gammas)):
                ax.annotate(
                    df['Source'][i], 
                    (df['L_IR (L_sun)'][i], L_gammas[i]),
                    xytext=(5, 0), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

        # Remove specific galaxies from fitting
        galaxies_to_exclude = ['Circinus Galaxy', 'NGC 2403', 'NGC 3424']
        fitting_mask = ~df['Source'].isin(galaxies_to_exclude)
        
        # Store L_gamma values corresponding to filtered galaxies
        L_gammas_filtered = L_gammas[fitting_mask]
        
        # Create filtered dataframe with L_gamma values
        df_for_fitting = df[fitting_mask].copy()
        df_for_fitting['L_gamma'] = L_gammas_filtered
        
        # Fit power law and plot
        self._fit_powerlaw(
            x_data=df_for_fitting['L_IR (L_sun)'],
            y_data=df_for_fitting['L_gamma'],
            color='orange',
            label_prefix='Obs',
            ax=ax
        )
        
        return ax

    def finalize(self, fig, ax, aperture=None):
        """Finalize plot with labels, scales, and save"""
        aperture = aperture or self.aperture
        #ax.text(0.05,0.95,f'Aperture: {aperture:.0f}', ha='left', va='top', transform=ax.transAxes)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$L_{\rm IR, ~8-1000 ~\mu m} ~(L_\odot)$')
        ax.set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        #ax.set_xmargin(0)
        ax.set_xlim(self.x_range[0], self.x_range[-1])
        ax.legend()
        plt.tight_layout()
        fig.savefig('LIR-Lgamma.png', dpi=300)
        
    def run(self, show_names=False):
        """Run the full analysis and plotting pipeline"""
        # Create figure
        fig, ax = plt.subplots(figsize=(5,4))
        
        # Process all galaxies
        
        self.make_table()
            
        # Create plots
        self.plot_sim(ax=ax)
        self.plot_obs(ax=ax, show_names=show_names)
        self.finalize(fig, ax)
        
        return fig, ax


def main(E_min=1*u.GeV, E_max=1000*u.GeV, show_names=False):
    print('Running LIR_Lgamma.py')
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
        'm10v_cd': [600],
        #'m09_cd': [600],
        'm11f_et_AlfvenMax': [600],
        'm11f_et_FastMax': [600],
        'm11f_sc_fcas50': [600],
    }
    
    # Create the plotter with our parameters
    plotter = LIR_Lgamma_Plot(
        galaxies=galaxies,
        E_min=E_min, 
        E_max=E_max,
        aperture=25*u.kpc
    )
    
    # Either run the full pipeline
    plotter.run(show_names=show_names)

if __name__ == '__main__':
    main(E_min=1*u.GeV, E_max=1000*u.GeV, show_names=False)
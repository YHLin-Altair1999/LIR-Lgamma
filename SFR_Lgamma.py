import numpy as np
import pandas as pd
from LIR_Lgamma import LIR_Lgamma_Plot
import astropy.units as u
import matplotlib.pyplot as plt
import os
from astropy.cosmology import Planck18, z_at_value, FlatLambdaCDM

class SFR_Lgamma_Plot(LIR_Lgamma_Plot):
    def __init__(self, 
                galaxies=None,
                E_min=1*u.GeV, E_max=1000*u.GeV,
                Lgamma_profile_folder='/tscc/lustre/ddn/scratch/yel051/tables/Lgamma_profiles',
                SFR_profile_folder='/tscc/lustre/ddn/scratch/yel051/tables/SFR_profiles',
                table_path='./tables/Lgamma_SFR.csv',
                aperture=25*u.kpc):
        """Initialize SFR_Lgamma_Plot with parameters for analysis and visualization"""
        super().__init__(galaxies, E_min, E_max, Lgamma_profile_folder, table_path, aperture)
        self.SFR_profile_folder = SFR_profile_folder
        self.x_range = np.logspace(-3, 3, 100)
        self.cosmo = Planck18
    
    def make_table(self, table_path=None, aperture=None):
        """Create or update table with gamma ray luminosities and SFR"""
        table_path = table_path or self.table_path
        aperture = aperture or self.aperture
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

        # The rest is the same as in the parent class
        if os.path.exists(table_path):
            existing_df = pd.read_csv(table_path)
            new_df = pd.DataFrame(data)
            df = pd.concat([existing_df, new_df]).drop_duplicates(subset='galaxy', keep='last')
        else:
            df = pd.DataFrame(data)
        df = df.sort_values(by='snap')
        df.to_csv(table_path, index=False)
    
    def plot_sim(self, table_path=None, ax=None):
        """Plot simulation data on the given axis"""
        table_path = table_path or self.table_path
        if ax is None:
            _, ax = plt.subplots(figsize=(5,4))
            
        df = pd.read_csv(table_path)
        for i in range(df.shape[0]):
            galaxy = df['galaxy'][i]
            ax.scatter(
                df['SFR (M_sun/yr)'][i], df['L_gamma (erg/s)'][i], 
                marker=self.get_marker(galaxy=galaxy), color=self.get_color(galaxy), 
                zorder=3, s=60, alpha=0.8, edgecolor='None')
        
        return ax
    
    def plot_obs(self, ax=None, E_min=1*u.GeV, E_max=1000*u.GeV, show_names=False):
        """Plot observational data on the given axis"""
        # Implement observation plotting using SFR data
        # You'll need to either:
        # 1. Load SFR values from observational data, or
        # 2. Convert L_IR to SFR using a standard relation
        
        # This is a placeholder implementation
        if ax is None:
            _, ax = plt.subplots(figsize=(5,4))
        
        # Load observations with SFR data
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
        
        epsilon = 0.79 # See Pfrommer 2017 Eq. 15
        df['SFR (M_sun/yr)'] = epsilon*1.7e-10*df['L_IR (L_sun)']
        
        # Plot observation data
        ax.scatter(
            df['SFR (M_sun/yr)'], L_gammas,
            label='Ambrosone et al. (2024)', color='gray', marker='^', 
            edgecolor='None', alpha=0.8, s=60, zorder=2
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
            x_data=df_for_fitting['SFR (M_sun/yr)'],
            y_data=df_for_fitting['L_gamma'],
            color='orange',
            label_prefix='Obs',
            ax=ax
        )
        
        return ax
    
    def finalize(self, fig, ax, aperture=None):
        """Finalize plot with labels, scales, and save"""
        aperture = aperture or self.aperture
        #ax.text(0.05, 0.95, f'Aperture: {aperture:.0f}', ha='left', va='top', transform=ax.transAxes)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\mathrm{SFR}~(M_\odot/\mathrm{yr})$')
        ax.set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.0f}-{self.E_max.to(u.GeV).value:.0f}~GeV}}}} ~{{\rm (erg/s)}}$')
        ax.set_xmargin(0)
        ax.legend()
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
        'm10v_cd': [600],
        'm11f_et_AlfvenMax': [600],
        'm11f_et_FastMax': [600],
        'm11f_sc_fcas50': [600]
    }
    
    plotter = SFR_Lgamma_Plot(
        galaxies=galaxies,
        E_min=E_min, 
        E_max=E_max,
        aperture=25*u.kpc
    )
    
    plotter.run(show_names=show_names)

if __name__ == '__main__':
    main(E_min=1*u.GeV, E_max=1000*u.GeV, show_names=False)
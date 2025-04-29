import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from My_Plugin.quantity import L_IR, L_gamma_yt, L_gamma_YHLin
from My_Plugin.LoadData import get_snap_path, get_center
import yt
import os
from glob import glob
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
import astropy.constants as c
from scipy import integrate
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
        self.galaxies = galaxies or {}
        self.E_min = E_min
        self.E_max = E_max
        self.Lgamma_profile_folder = Lgamma_profile_folder
        self.table_path = table_path
        self.aperture = aperture

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

    def make_table(self, galaxy, snaps, table_path=None, aperture=None):
        """Create or update table with gamma ray and IR luminosities"""
        table_path = table_path or self.table_path
        aperture = aperture or self.aperture
        data = []
        
        for snap in snaps:
            fname = os.path.join(self.Lgamma_profile_folder, f'Lgamma_profile_{galaxy}_snap{snap:03d}.npy')
            profile = np.load(fname)
            #print(profile)
            Lgamma = np.sum(profile[:,1])
            data.append({
                'galaxy': galaxy, 
                'snap': snap, 
                'L_gamma (erg/s)': Lgamma,
                #'L_gamma (erg/s)': self.calculate_Lgamma(galaxy, snap, mode='YHLin', aperture=aperture).to('erg/s').value, 
                'L_IR (L_sun)': self.calculate_LIR(galaxy, snap).to('L_sun').value
                })

        if os.path.exists(table_path):
            existing_df = pd.read_csv(table_path)
            new_df = pd.DataFrame(data)
            df = pd.concat([existing_df, new_df]).drop_duplicates(subset='galaxy', keep='last')
        else:
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

    def plot_sim(self, table_path=None, ax=None):
        """Plot simulation data on the given axis"""
        table_path = table_path or self.table_path
        if ax is None:
            _, ax = plt.subplots(figsize=(5,4))
            
        df = pd.read_csv(table_path)
        for i in range(df.shape[0]):
            galaxy = df['galaxy'][i]
            ax.scatter(
                df['L_IR (L_sun)'][i], df['L_gamma (erg/s)'][i], 
                #label=df['galaxy'][i], 
                marker=self.get_marker(galaxy=galaxy), color=self.get_color(galaxy), zorder=3, s=60, alpha=0.8, edgecolor='None')
        return ax

    def plot_obs(self, ax=None, E_min=None, E_max=None):
        """Plot observational data on the given axis"""
        if ax is None:
            _, ax = plt.subplots(figsize=(5,4))
        E_min = E_min or self.E_min
        E_max = E_max or self.E_max
        
        def luminosity(E, phi_0, gamma, D_L):
            z = z_at_value(Planck18.luminosity_distance, D_L)
            dFdE = E * phi_0*(E/u.GeV)**(-gamma)
            flux = integrate.simpson(dFdE.value, x=E.value) * (E.unit * dFdE.unit)
            L = 4*np.pi * D_L**2 * (1+z)**(gamma-2) * flux
            return L.to('erg/s').value
        
        obs_path = './obs_data/Ambrosone_2024.csv'
        df = pd.read_csv(obs_path)
        df['L_IR (L_sun)'] = np.array(df['LIR'])*1e10
        df['z'] = z_at_value(Planck18.luminosity_distance, np.array(df['DL'])*u.Mpc)
        L_gammas = np.array([
            luminosity(
                np.logspace(np.log10(E_min.to(u.GeV).value),np.log10(E_max.to(u.GeV).value),100) * u.GeV, 
                df['Phi0'][i] * 1e-12 * u.cm**-2 * u.s**-1 * u.MeV**-1, 
                df['gamma'][i], 
                df['DL'][i]*u.Mpc
                ) for i in range(len(df['z']))
            ])
        
        #for i in range(len(L_gammas)):
            #print(df['Source'][i], L_gammas[i])

        ax.scatter(
            df['L_IR (L_sun)'], L_gammas,
            label='Ambrosone et al. (2024)', color='gray', marker='^', edgecolor='None', alpha=0.8, s=60, zorder=2
            )

        # scaling relation from Ackermann 2012
        LIR_A12 = np.logspace(6, 13, 100)*u.L_sun
        alpha, beta = 1.09, 39.19
        alpha_error, beta_error = 0.1, 0.1
        Lgamma_A12_bestfit = 10**(alpha*np.log10(LIR_A12/u.L_sun/1e10) + beta) # in unit of erg/s, A12 eq. 4
        ax.plot(LIR_A12.to('L_sun').value, Lgamma_A12_bestfit, label=r'A12 scaling relation ($0.1-100$ GeV)', color='orange', zorder=1)

        Lgamma_1 = 10**((alpha+alpha_error)*np.log10(LIR_A12/u.L_sun/1e10) + (beta+beta_error))
        Lgamma_2 = 10**((alpha-alpha_error)*np.log10(LIR_A12/u.L_sun/1e10) + (beta-beta_error))
        Lgamma_3 = 10**((alpha+alpha_error)*np.log10(LIR_A12/u.L_sun/1e10) + (beta-beta_error))
        Lgamma_4 = 10**((alpha-alpha_error)*np.log10(LIR_A12/u.L_sun/1e10) + (beta+beta_error))
        stacked = np.vstack((Lgamma_1, Lgamma_2, Lgamma_3, Lgamma_4))
        ax.fill_between(LIR_A12.to('L_sun').value, np.min(stacked, axis=0), np.max(stacked, axis=0), color='orange', alpha=0.5, zorder=1)

        return ax

    def finalize(self, fig, ax, aperture=None):
        """Finalize plot with labels, scales, and save"""
        aperture = aperture or self.aperture
        ax.text(0.05,0.95,f'Aperture: {aperture:.0f}', ha='left', va='top', transform=ax.transAxes)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$L_{\rm IR, ~8-1000 ~\mu m} ~(L_\odot)$')
        ax.set_ylabel(rf'$L_{{\gamma, ~{{\rm {self.E_min.to(u.GeV).value:.1f}-{self.E_max.to(u.GeV).value:.1f}~GeV}}}} ~{{\rm (erg/s)}}$')
        ax.set_xmargin(0)
        ax.legend()
        plt.tight_layout()
        fig.savefig('LIR-Lgamma.png', dpi=300)
        
    def run(self):
        """Run the full analysis and plotting pipeline"""
        # Create figure
        fig, ax = plt.subplots(figsize=(5,4))
        
        # Process all galaxies
        for galaxy, snaps in self.galaxies.items():
            self.make_table(galaxy, snaps)
            
        # Create plots
        self.plot_sim(ax=ax)
        self.plot_obs(ax=ax)
        self.finalize(fig, ax)
        
        return fig, ax


def main(E_min=1*u.GeV, E_max=1000*u.GeV):
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
        'm11f_sc_fcas50': [600]
    }
    
    # Create the plotter with our parameters
    plotter = LIR_Lgamma_Plot(
        galaxies=galaxies,
        E_min=E_min, 
        E_max=E_max,
        aperture=25*u.kpc
    )
    
    # Either run the full pipeline
    plotter.run()
    
    # Or execute steps individually
    # fig, ax = plt.subplots(figsize=(5,4))
    # for galaxy, snaps in galaxies.items():
    #     plotter.make_table(galaxy, snaps)
    # plotter.plot_sim(ax=ax)
    # plotter.plot_obs(ax=ax)
    # plotter.finalize(fig, ax)

if __name__ == '__main__':
    main(E_min=0.11*u.GeV, E_max=100*u.GeV)
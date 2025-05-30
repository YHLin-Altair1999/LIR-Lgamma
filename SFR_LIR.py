import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import os
from scipy.optimize import curve_fit
from LIR_Lgamma import LIR_Lgamma_Plot

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
})

class SFR_LIR_Plot:
    def __init__(self, 
                galaxies=None,
                SFR_profile_folder='/tscc/lustre/ddn/scratch/yel051/tables/SFR_profiles',
                table_path='./tables/SFR_LIR.csv',
                aperture=25*u.kpc):
        """Initialize SFR_LIR_Plot with parameters for analysis and visualization"""
        self.galaxies = galaxies or {}
        self.SFR_profile_folder = SFR_profile_folder
        self.table_path = table_path
        self.aperture = aperture
        # Create LIR_Lgamma_Plot instance to use its LIR calculation method
        self.lir_calculator = LIR_Lgamma_Plot()
        
    def make_table(self, galaxy, snaps, table_path=None, aperture=None):
        """Create a new table with SFR and LIR values"""
        table_path = table_path or self.table_path
        aperture = aperture or self.aperture
        data = []
        
        for snap in snaps:
            # Get SFR from profile
            fname = os.path.join(self.SFR_profile_folder, f'SFR_profile_{galaxy}_snap{snap:03d}.npy')
            profile = np.load(fname)
            SFR = np.sum(profile[:,1])*u.Msun/u.yr
            
            # Get LIR using the method from LIR_Lgamma_Plot
            LIR = self.lir_calculator.calculate_LIR(galaxy, snap).to('L_sun').value
            
            # Store data
            data.append({
                'galaxy': galaxy, 
                'snap': snap, 
                'SFR (M_sun/yr)': SFR.to('Msun/yr').value,
                'L_IR (L_sun)': LIR
            })

        # Always create a new DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values(by=['galaxy', 'snap'])
        
        return df
    
    def get_marker(self, galaxy: str) -> str:
        """Return marker style based on galaxy type"""
        gal_type = ''.join(galaxy.split('_')[1:])
        match gal_type:
            case 'cd':
                marker = 'o'
            case _:
                marker = 's'
        return marker

    def get_color(self, galaxy: str) -> str:
        """Return color based on galaxy type"""
        gal_type = galaxy.split('_')[1]
        match gal_type:
            case 'cd':
                color = 'C0'
            case 'et':
                color = 'C1'
            case 'sc':
                color = 'C2'
        return color
        
    def plot(self, df=None, table_path=None, fit=True):
        """Create a scatter plot of SFR vs LIR"""
        if df is None:
            # If no DataFrame is provided, generate it from all galaxies
            all_data = []
            for galaxy, snaps in self.galaxies.items():
                galaxy_df = self.make_table(galaxy, snaps)
                all_data.append(galaxy_df)
            
            if all_data:
                df = pd.concat(all_data)
            else:
                print("No galaxies specified. Cannot create plot.")
                return None, None
        
        # Create the plot
        fig, axes = plt.subplots(
            figsize=(5, 5), nrows=2, ncols=1, sharex=True, sharey=False,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
            )
        

        ax = axes[0]

        # Plot each galaxy
        for i in range(df.shape[0]):
            galaxy = df['galaxy'].iloc[i]
            ax.scatter(
                df['SFR (M_sun/yr)'].iloc[i], 
                df['L_IR (L_sun)'].iloc[i], 
                marker=self.get_marker(galaxy=galaxy), 
                color=self.get_color(galaxy), 
                s=60, alpha=0.8, edgecolor='None',
                #label=galaxy if i == df['galaxy'].tolist().index(galaxy) else ""
            )
            # Add galaxy name annotation beside each data point
            ax.annotate(
                galaxy, 
                (df['SFR (M_sun/yr)'].iloc[i], df['L_IR (L_sun)'].iloc[i]),
                xytext=(5, 0),  # Small offset from the point
                textcoords='offset points',
                fontsize=8,
                color=self.get_color(galaxy)
            )
        
        def SFR2LIR(SFR, epsilon=0.79):
            """Convert SFR to LIR using the Kennicutt relation"""
            return (SFR/(u.Msun/u.yr))/(epsilon*1.7e-10)*u.L_sun

        SFR_range = np.logspace(-3, 1, 100)*u.Msun/u.yr
        LIR_Kennicutt = SFR2LIR(SFR_range)

        ax.plot(
            SFR_range.to('M_sun/yr').value, 
            LIR_Kennicutt.to('L_sun').value, 
            color='k', linestyle='--', label=r'Kennicutt (1998)'
            )

        # Using Bonato et al. (2024) equ. 2
        def SFR2LIR_Bonato(SFR):
            return (SFR.to('M_sun*yr**-1').value*10**(7.81))**(1/0.8)*u.L_sun
        
        LIR_Bonato = SFR2LIR_Bonato(SFR_range)
        ax.plot(
            SFR_range.to('M_sun/yr').value,
            LIR_Bonato.to('L_sun').value,
            color='k', linestyle='dotted', label=r'Bonato et al. (2024)'
            )
        
        # Set axis labels and scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(SFR_range[0].value, SFR_range[-1].value)
        #ax.set_ylim(1e5, 1e11)
        ax.set_ylabel(r'$L_{\rm IR, ~8-1000 ~\mu m} ~(L_\odot)$')
        
        # Add other plot elements
        handles, labels = ax.get_legend_handles_labels()
        # Combine duplicate entries
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
        
        ax2 = axes[1]
        for i in range(df.shape[0]):
            galaxy = df['galaxy'].iloc[i]
            ax2.scatter(
                df['SFR (M_sun/yr)'].iloc[i], 
                df['L_IR (L_sun)'].iloc[i]/SFR2LIR(df['SFR (M_sun/yr)'].iloc[i]*u.M_sun/u.yr).to('L_sun').value, 
                marker=self.get_marker(galaxy=galaxy), 
                color=self.get_color(galaxy), 
                s=60, alpha=0.8, edgecolor='None',
                #label=galaxy if i == df['galaxy'].tolist().index(galaxy) else ""
            )
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$\mathrm{SFR}~(M_\odot/\mathrm{yr})$')
        ax2.axhline(1, color='k', linestyle='--', label=r'$L_{\rm IR}/L_{\rm IR, Kennicutt} = 1$')
        ax2.set_ylabel(r'$L_{\rm IR}/L_{\rm IR, Kennicutt}$')

        plt.tight_layout()
        fig.savefig('SFR-LIR.png', dpi=300)
        
        return fig, ax
    
    def run(self):
        """Run the full analysis and plotting pipeline by creating a new dataframe"""
        # Process all galaxies and create table
        all_data = []
        for galaxy, snaps in self.galaxies.items():
            df = self.make_table(galaxy, snaps)
            all_data.append(df)
        
        # Combine data if multiple galaxies were processed
        if all_data:
            combined_df = pd.concat(all_data)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.table_path), exist_ok=True)
            # Save the DataFrame to CSV
            combined_df.to_csv(self.table_path, index=False, mode='w')  # 'w' mode to overwrite
            # Plot the data
            self.plot(combined_df)
        else:
            print("No galaxies specified. Cannot create plot.")

def main():
    print('Running SFR_LIR.py')
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
    
    plotter = SFR_LIR_Plot(galaxies=galaxies)
    plotter.run()

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import os
from My_Plugin.LIR_Lgamma import LIR_Lgamma_Plot

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
})

class SFR_LIR_Plot(LIR_Lgamma_Plot):
    def __init__(self, 
                galaxies=None,
                SFR_folder='/tscc/lustre/ddn/scratch/yel051/tables/SFR',
                sim_table_path='./tables/SFR_LIR.csv',
                sed_base_path='/tscc/lustre/ddn/scratch/yel051/SKIRT/output/',
                aperture=25*u.kpc,
                show_sim_gal_name=False,
                plot_sim_residuals=True,
                x_range=np.logspace(-3, 1, 100),  # SFR range for plotting
                y_range=np.logspace(6.1, 11, 100),  # LIR range for plotting
                output_filename='SFR-LIR.png',
                residual_dynamic_range=10,
                plottype_annotation=None,
                ):
        """Initialize SFR_LIR_Plot with parameters for analysis and visualization"""
        # Call parent constructor with appropriate parameters
        super().__init__(
            galaxies=galaxies,
            sim_table_path=sim_table_path,
            sed_base_path=sed_base_path,
            show_sim_gal_name=show_sim_gal_name,
            output_filename=output_filename
        )
        
        # SFR-specific configuration
        self.SFR_folder = SFR_folder
        self.aperture = aperture
        self.x_range = x_range
        self.y_range = y_range
        self.residual_dynamic_range = residual_dynamic_range
        self.plottype_annotation = plottype_annotation
        
        # Controls specific to SFR-LIR plotting
        self.plot_sim_residuals = plot_sim_residuals
        
        # Kennicutt relation parameters
        self.epsilon = 0.79  # correcting factor for IMF
        
    def make_sim_table(self):
        """Create or update table with SFR and IR luminosities"""
        data = []
        
        for galaxy, snaps in self.galaxies.items():
            for snap in snaps:
                # Get SFR from profile
                fname = os.path.join(self.SFR_folder, f'SFR_{galaxy}_snap{snap:03d}.npy')
                SFR, SFR_err = np.load(fname)*u.Msun/u.yr
                #print(f"Galaxy: {galaxy}, Snap: {snap}, SFR: {SFR.to('Msun/yr').value:.2e} Msun/yr")
                
                # Get LIR using the inherited method
                LIR = self.calculate_LIR(galaxy, snap).to('L_sun').value
                
                # Store data
                data.append({
                    'galaxy': galaxy, 
                    'snap': snap, 
                    'SFR (M_sun/yr)': SFR.to('Msun/yr').value,
                    'SFR_err (M_sun/yr)': SFR_err.to('Msun/yr').value,
                    'L_IR (L_sun)': LIR
                })

        # Always create a new DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values(by=['galaxy', 'snap'])
        
        # Save the table
        os.makedirs(os.path.dirname(self.sim_table_path), exist_ok=True)
        df.to_csv(self.sim_table_path, index=False)
        self.sim_table = df  # Use the parent class attribute name
        
        return df
    
    def SFR2LIR_Kennicutt(self, SFR):
        """Convert SFR to LIR using the Kennicutt relation"""
        SFR_array = np.atleast_1d(SFR)
        return (SFR_array/(u.Msun/u.yr))/(self.epsilon*1.7e-10)*u.L_sun

    def SFR2LIR_Bonato(self, SFR):
        """Convert SFR to LIR using Bonato et al. (2024) equ. 2"""
        SFR_array = np.atleast_1d(SFR)
        return (SFR_array.to('M_sun*yr**-1').value*10**(7.81))**(1/0.8)*u.L_sun
    def plot_main(self, ax=None, sim_table_path=None):
        """Plot simulation data on the main panel"""
        sim_table_path = sim_table_path or self.sim_table_path
        ax = ax or self.axes[0]
        
        # Load or use existing simulation data
        if hasattr(self, 'sim_table') and self.sim_table is not None:
            df = self.sim_table
        else:
            df = pd.read_csv(sim_table_path)
        
        # Plot simulation data points
        for i in range(df.shape[0]):
            galaxy = df['galaxy'].iloc[i]
            ax.errorbar(
                df['SFR (M_sun/yr)'].iloc[i], 
                df['L_IR (L_sun)'].iloc[i],
                xerr=df['SFR_err (M_sun/yr)'].iloc[i],
                marker=self.get_marker(galaxy=galaxy), 
                color=self.get_color(galaxy), 
                markersize=8, alpha=0.8, markeredgecolor='None',
                ecolor=self.get_color(galaxy), elinewidth=1, capsize=3,
                linestyle='none'  # Only show markers and error bars, no connecting lines
            )
            # Add galaxy name annotation beside each data point
            if self.show_sim_gal_name:
                ax.annotate(
                    galaxy, 
                    (df['SFR (M_sun/yr)'].iloc[i], df['L_IR (L_sun)'].iloc[i]),
                    xytext=(5, 0),  # Small offset from the point
                    textcoords='offset points',
                    fontsize=8,
                    color=self.get_color(galaxy)
                )
        
        # Plot theoretical relations
        SFR_range = self.x_range * u.Msun/u.yr
        LIR_Kennicutt = self.SFR2LIR_Kennicutt(SFR_range)
        LIR_Bonato = self.SFR2LIR_Bonato(SFR_range)

        ax.plot(
            SFR_range.to('M_sun/yr').value, 
            LIR_Kennicutt.to('L_sun').value, 
            color='k', linestyle='--', label=r'Kennicutt (1998)'
        )

        ax.plot(
            SFR_range.to('M_sun/yr').value,
            LIR_Bonato.to('L_sun').value,
            color='k', linestyle='dotted', label=r'Bonato et al. (2024)'
        )
        
        # Configure the main axis
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(self.x_range[0], self.x_range[-1])
        ax.set_ylim(self.y_range[0], self.y_range[-1])
        ax.set_ylabel(r'$L_{\rm IR, ~8-1000 ~\mu m} ~(L_\odot)$')
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        # Combine duplicate entries
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
        
        return ax

    def plot_residuals(self, ax):
        """Plot residuals (data/Kennicutt relation) in the lower panel using a log scale"""
        # Load simulation data
        if hasattr(self, 'sim_table') and self.sim_table is not None:
            df = self.sim_table
        else:
            df = pd.read_csv(self.sim_table_path)
        
        # Plot simulation residuals relative to Kennicutt relation
        for i in range(df.shape[0]):
            galaxy = df['galaxy'].iloc[i]
            # Calculate the ratio and propagate error
            kennicutt_lir = self.SFR2LIR_Kennicutt(df['SFR (M_sun/yr)'].iloc[i]*u.M_sun/u.yr).to('L_sun').value
            ratio = df['L_IR (L_sun)'].iloc[i] / kennicutt_lir
            
            # Error propagation for the ratio (assuming LIR has no error for simplicity)
            # ratio_err ≈ (dLIR/dSFR) * SFR_err / kennicutt_lir
            # For Kennicutt relation: dLIR/dSFR is proportional to 1/SFR, so error scales as SFR_err/SFR
            ratio_err = ratio * (df['SFR_err (M_sun/yr)'].iloc[i] / df['SFR (M_sun/yr)'].iloc[i])
            
            ax.errorbar(
                df['SFR (M_sun/yr)'].iloc[i], 
                ratio,
                xerr=df['SFR_err (M_sun/yr)'].iloc[i],
                yerr=ratio_err,
                marker=self.get_marker(galaxy=galaxy), 
                color=self.get_color(galaxy), 
                markersize=8, alpha=0.8, markeredgecolor='None',
                ecolor=self.get_color(galaxy), elinewidth=1, capsize=3,
                linestyle='none'
            )
        
        # Add horizontal reference lines
        ax.axhline(1, color='k', linestyle='--', label=r'$L_{\rm IR}/L_{\rm IR, Kennicutt} = 1$')
        
        # Configure the residuals axis
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$L_{\rm IR}/L_{\rm IR, Kennicutt}$')
        ax.set_ylim(1/self.residual_dynamic_range, self.residual_dynamic_range)

    def finalize(self, fig, axes):
        """Finalize plot with labels, scales, and save"""
        # Configure lower panel if it exists
        if axes[1] is not None:
            axes[1].set_xlabel(r'$\mathrm{SFR}~(M_\odot/\mathrm{yr})$')
        else:
            axes[0].set_xlabel(r'$\mathrm{SFR}~(M_\odot/\mathrm{yr})$')
        
        if self.plottype_annotation != None:
            # Add plot type annotation
            axes[0].text(
                0.95, 0.05, self.plottype_annotation, transform=axes[0].transAxes, 
                va='bottom', ha='right'
            )
        plt.tight_layout()
        print(f'Saving figure to {self.output_filename}')
        fig.savefig(self.output_filename, dpi=300)

    def run(self):
        """Run the full analysis and plotting pipeline"""
        # Create figure with two panels (or one if residuals not needed)
        if self.plot_sim_residuals:
            self.fig, self.axes = plt.subplots(
                figsize=(4, 4), 
                nrows=2, ncols=1, 
                sharex=True, sharey=False,
                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
            )
        else:
            self.fig, ax = plt.subplots(figsize=(4, 4))
            self.axes = [ax, None]

        # Process data
        self.make_sim_table()
        
        # Create plots for the main panel
        self.plot_main(ax=self.axes[0])
        
        # Handle residuals if needed
        if self.plot_sim_residuals and self.axes[1] is not None:
            # Hide x tick labels for top panel
            plt.setp(self.axes[0].get_xticklabels(), visible=False)
            
            # Plot residuals
            self.plot_residuals(self.axes[1])
        
        # Finalize the plot
        self.finalize(self.fig, self.axes)
        
        return self.fig, self.axes

def main():
    """
    Usage example for SFR_LIR_Plot class
    """
    print('Running SFR_LIR.py')
    galaxies = {
        'm12i_cd': [600],
    }
    
    # Create the plotter with our parameters
    plotter = SFR_LIR_Plot(
        galaxies=galaxies,
        show_sim_gal_name=True,
        plot_sim_residuals=True,
        # sed_base_path='/custom/path/to/SKIRT/output/'  # Optional: specify custom SED path
    )
    
    plotter.run()

if __name__ == '__main__':
    main()

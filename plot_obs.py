#!/usr/bin/env python3
"""
Script to load and plot results from fit_obs_data.py
Uses pickle format for machine-readable output
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Configure plot settings
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif"
})

def plot_from_pickle(pickle_file='fit_results.pkl', output_file='recreated_plot.png', 
                    interval_type=None):
    """Load results from pickle and recreate the plot"""
    
    # Load results
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 4))
    
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
    
    # Plot included points with error bars
    ax.errorbar(
        x_data[included_mask], y_data[included_mask], 
        yerr=[yerr[0][included_mask], yerr[1][included_mask]],
        fmt='^', color='gray', ecolor='gray', capsize=3,
        markersize=7, label='Ambrosone et al. (2024)', alpha=0.7, zorder=3
    )
    
    # Plot excluded points (hollow markers)
    if np.any(~included_mask):
        ax.scatter(
            x_data[~included_mask], y_data[~included_mask],
            facecolors='none', marker='^', 
            edgecolor='gray', alpha=0.7, s=49, zorder=2
        )
    
    # Plot fit line and confidence interval
    x_fit = np.logspace(np.log10(x_data.min()) - 0.5, np.log10(x_data.max()) + 0.5, 1000)
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
    ax.plot(x_fit, y_fit, '--', color='gray', zorder=4)
    
    # Plot confidence/prediction interval
    interval_label = f'$1\\sigma$ {interval_type} interval'
    ax.fill_between(x_fit, y_lower, y_upper, color='gray', alpha=0.2, zorder=2)
    
    # Configure axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$L_{\mathrm{IR}}$ ($L_{\odot}$)')
    ax.set_ylabel(r'$L_{\gamma}$ (erg s$^{-1}$)')
    
    # Add correlation coefficient
    r_value = fit_params['correlation_coefficient']
    p_value = fit_params['p_value']
    ax.text(0.05, 0.95, f'$r = {r_value:.3f}$\n$p = {p_value:.3e}$', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Plot recreated from {pickle_file} and saved as {output_file}")
    
    # Print fit results
    print(f"\nFit Results:")
    print(f"A = {A:.2e} ± {fit_params['A_error']:.2e}")
    print(f"B = {B:.3f} ± {fit_params['B_error']:.3f}")
    print(f"r = {r_value:.3f}")
    print(f"p = {p_value:.2e}")
    print(f"Interval type: {interval_type}")
    
    return results

def main():
    """Load and plot the results"""
    
    try:
        print("Loading results from pickle file...")
        results = plot_from_pickle()
        
        # Example: Create another plot with different interval type
        current_type = results['fit_info']['interval_type']
        other_type = 'confidence' if current_type == 'prediction' else 'prediction'
        
        print(f"\nCreating additional plot with {other_type} interval...")
        plot_from_pickle(output_file=f'plot_{other_type}_interval.png', 
                        interval_type=other_type)
        
        print("\nAll plots created successfully!")
        
    except FileNotFoundError:
        print("Error: fit_results.pkl not found.")
        print("Please run fit_obs_data.py first to generate the results file.")

if __name__ == '__main__':
    main()

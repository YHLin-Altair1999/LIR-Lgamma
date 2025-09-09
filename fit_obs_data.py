import json
import pickle
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# Constants
rng = np.random.default_rng(42)
cosmo = Planck18
PHI_UNIT = u.Unit('1e-12 cm^-2 s^-1 MeV^-1')
N_SAMPLES = 1000

def luminosity(phi_0, gamma, D_L, E_min=1*u.GeV, E_max=1000*u.GeV):
    """Calculate gamma-ray luminosity given flux parameters"""
    z = z_at_value(cosmo.luminosity_distance, D_L)
    flux = phi_0*(u.GeV)**2/(2-gamma)*((E_max/u.GeV)**(2-gamma)-(E_min/u.GeV)**(2-gamma))
    return (4*np.pi * D_L**2 * (1+z)**(gamma-2) * flux).to('erg/s').value

def calculate_gamma_luminosities(df):
    """Process dataframe to calculate gamma-ray luminosities and uncertainties"""
    df['z'] = z_at_value(cosmo.luminosity_distance, df['DL'].values * u.Mpc)
    df['L_IR (L_sun)'] = df['LIR'] * 1e10 * u.L_sun
    
    results = []
    for _, row in df.iterrows():
        DL = row['DL'] * u.Mpc
        # Monte Carlo sampling for error propagation
        phis = rng.normal(loc=row['Phi0'], scale=row['Phi0_plus'], size=N_SAMPLES)
        gammas = rng.normal(loc=row['gamma'], scale=row['gamma_plus'], size=N_SAMPLES)
        Ls = luminosity(phis*PHI_UNIT, gammas, DL)
        
        # Calculate central value and percentiles
        L_gamma = luminosity(row['Phi0']*PHI_UNIT, row['gamma'], DL)
        results.append((L_gamma, np.percentile(Ls, 16), np.percentile(Ls, 84)))
    
    # Unpack results
    df['L_gamma (erg/s)'], df['L_gamma_err_minus'], df['L_gamma_err_plus'] = zip(*results)
    return df

def linear_fit(x, alpha, beta):
    """Linear fitting function for log-log space"""
    return alpha * x + beta

def create_confidence_function(actual_intercept, slope, x_fit_data, y_fit_data):
    """
    Create a confidence/prediction interval function that can be pickled
    """
    log_x_data = np.log10(x_fit_data)
    n = len(log_x_data)
    x_mean = np.mean(log_x_data)
    sum_x_sq = np.sum((log_x_data - x_mean)**2)
    
    # Calculate residual standard error from the fit
    log_y_pred = actual_intercept + slope * log_x_data
    log_y_actual = np.log10(y_fit_data)
    residuals = log_y_actual - log_y_pred
    residual_std = np.sqrt(np.sum(residuals**2) / (n - 2)) if n > 2 else 0.1
    
    def calculate_confidence_interval(x_plot, interval_type='prediction'):
        """
        Calculate confidence or prediction interval at given x points
        """
        log_x_plot = np.log10(x_plot)
        
        # Standard error calculation depends on interval type
        if interval_type == 'confidence':
            # Confidence interval: uncertainty in the fitted line only
            se = residual_std * np.sqrt(1/n + (log_x_plot - x_mean)**2 / sum_x_sq)
        elif interval_type == 'prediction':
            # Prediction interval: includes both fit uncertainty AND intrinsic scatter
            se = residual_std * np.sqrt(1 + 1/n + (log_x_plot - x_mean)**2 / sum_x_sq)
        else:
            raise ValueError("interval_type must be 'confidence' or 'prediction'")
        
        # Convert back to linear space
        log_y_plot = actual_intercept + slope * log_x_plot
        y_upper = 10**(log_y_plot + se)
        y_lower = 10**(log_y_plot - se)
        
        return y_lower, y_upper
    
    return calculate_confidence_interval


def fit_loglog_relation(df, exclude_galaxies=None):
    """Fit log-log relation between L_IR and L_gamma"""
    
    # Remove excluded galaxies
    if exclude_galaxies is None:
        exclude_galaxies = []
    
    # Create mask for included galaxies
    mask = ~df['Source'].isin(exclude_galaxies)
    df_fit = df[mask].copy()
    
    # Extract data for fitting
    x_fit = df_fit['L_IR (L_sun)'].values
    y_fit = df_fit['L_gamma (erg/s)'].values
    
    print(f"Fitting {len(x_fit)} galaxies (excluded {len(exclude_galaxies)})")
    
    # Fit in log-log space  
    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)
    
    # Perform linear regression in log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # Calculate covariance matrix for error estimation
    n = len(log_x)
    x_mean = np.mean(log_x)
    sum_x_sq = np.sum((log_x - x_mean)**2)
    
    # Calculate coefficient errors
    residuals = log_y - (intercept + slope * log_x)
    mse = np.sum(residuals**2) / (n - 2) if n > 2 else 1.0
    slope_err = np.sqrt(mse / sum_x_sq)
    intercept_err = np.sqrt(mse * (1/n + x_mean**2/sum_x_sq))
    
    # Convert to power law form: y = A * x^B
    B = slope
    actual_intercept = intercept
    A = 10**actual_intercept
    
    # Error propagation for A and B
    A_err = A * np.log(10) * intercept_err
    B_err = slope_err
    
    # Create fit function for later use
    def fit_function(x_val):
        return A * (x_val)**B
    
    # Create fit result dictionary (excluding unpicklable functions)
    fit_result = {
        'alpha': B,
        'alpha_error': B_err,
        'beta': actual_intercept,
        'beta_error': intercept_err,
        'ref_point': 1.0,
        'A': A,
        'A_err': A_err,
        'B': B,
        'B_err': B_err,
        'r_value': r_value,
        'p_value': p_value,
        # Parameters for recreating confidence intervals
        'confidence_params': {
            'actual_intercept': actual_intercept,
            'slope': slope,
            'x_fit_data': x_fit,
            'y_fit_data': y_fit,
            'residual_std': np.sqrt(np.sum((log_y - (intercept + slope * log_x))**2) / (n - 2)) if n > 2 else 0.1,
            'n_points': n,
            'x_mean': x_mean,
            'sum_x_sq': sum_x_sq
        }
    }
    
    return (A, B), (A_err, B_err), (r_value, p_value), fit_result

def run(interval_type='prediction'):
    """
    Run the fitting pipeline and save results to pickle file
    
    Parameters:
    -----------
    interval_type : str
        'confidence' for confidence interval or 'prediction' for prediction interval
    """
    df = pd.read_csv('./obs_data/Ambrosone_2024.csv')
    df = calculate_gamma_luminosities(df)
    
    # Define galaxies to exclude from fitting (as in LIR_Lgamma.py)
    galaxies_to_exclude = ['Circinus', 'NGC 2403', 'NGC 3424']
    
    # Extract data for fitting
    x_data = df['L_IR (L_sun)'].values
    y_data = df['L_gamma (erg/s)'].values
    # Use average of asymmetric errors
    y_err = (df['L_gamma_err_minus'].values + df['L_gamma_err_plus'].values) / 2
    source_names = df['Source'].values
    
    # Perform fitting with exclusions
    params, perr, stats_info, fit_result = fit_loglog_relation(
        df, exclude_galaxies=galaxies_to_exclude
    )
    A, B = params
    A_err, B_err = perr
    r_value, p_value = stats_info
    
    # Prepare machine-readable results (pickle format only)
    results_dict = {
        'fit_parameters': {
            'A': float(A),
            'A_error': float(A_err),
            'B': float(B),
            'B_error': float(B_err),
            'correlation_coefficient': float(r_value),
            'p_value': float(p_value)
        },
        'data': {
            'L_IR_Lsun': x_data,
            'L_gamma_erg_per_s': y_data,
            'L_gamma_errors': y_err,
            'source_names': source_names,
            'excluded_galaxies': galaxies_to_exclude
        },
        'fit_info': {
            'equation': 'L_gamma = A * L_IR^B',
            'units': {'L_IR': 'L_sun', 'L_gamma': 'erg/s'},
            'interval_type': interval_type,
            'energy_range': {'E_min_GeV': 1.0, 'E_max_GeV': 1000.0}
        },
        'fit_result': fit_result,  # Includes confidence_params for recreating intervals
        'dataframe': df,
        'usage_instructions': {
            'power_law_function': 'To recreate the power law fit: y = A * x^B where A and B are in fit_parameters',
            'confidence_intervals': 'Use confidence_params in fit_result to recreate confidence/prediction intervals',
            'plotting_example': 'x_fit = np.logspace(...); y_fit = A * x_fit**B',
            'note': 'Functions are not pickled to avoid serialization issues. Recreate as needed using stored parameters.'
        }
    }
    
    # Save as pickle (preserves all Python objects including functions)
    with open('./obs_data/fit_results.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"Results saved as fit_results.pkl")
    
    return results_dict

if __name__ == '__main__':
    # You can change this to 'confidence' or 'prediction'
    run(interval_type='prediction')

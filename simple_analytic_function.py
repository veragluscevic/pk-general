#!/usr/bin/env python3
"""
Simplified analytic transfer function for interacting dark matter.

Based on the physics of IDM, this provides a more robust functional form
that captures the essential features with fewer parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

def simple_idm_transfer_function(k, A, k_c, n):
    """
    Simple 3-parameter analytic form for IDM transfer functions.
    
    This captures the essential physics:
    - Suppression at high k due to free-streaming/interactions
    - Scale-dependent transition
    - Minimal parameters for robustness
    
    Parameters:
    -----------
    k : array_like
        Wavenumber in h/Mpc
    A : float
        Amplitude of deviation (can be positive or negative)
    k_c : float
        Characteristic transition scale in h/Mpc
    n : float
        Power-law index for the suppression
        
    Returns:
    --------
    T_norm : array_like
        Normalized transfer function T_IDM(k) / T_ΛCDM(k)
        
    Notes:
    ------
    The function is: T_norm(k) = 1 + A * (k/k_c)^n / (1 + (k/k_c)^2)
    
    This form:
    - Starts at 1 + A for k << k_c
    - Transitions at k ≈ k_c  
    - Approaches 1 + A * (k/k_c)^(n-2) for k >> k_c
    - Can model both enhancement (A > 0) and suppression (A < 0)
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    ratio = k_safe / k_c
    numerator = ratio**n
    denominator = 1 + ratio**2
    
    T_norm = 1 + A * numerator / denominator
    
    return T_norm

def exponential_suppression_function(k, A, k_c, alpha):
    """
    Alternative form with exponential suppression.
    
    T_norm(k) = 1 + A * exp(-(k/k_c)^alpha)
    
    This is good for modeling sharp cutoffs.
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    ratio = k_safe / k_c
    suppression = np.exp(-ratio**alpha)
    
    T_norm = 1 + A * suppression
    
    return T_norm

def power_law_suppression_function(k, A, k_c, n, m):
    """
    Power-law suppression form.
    
    T_norm(k) = 1 + A * (k/k_c)^n / (1 + (k/k_c)^m)
    
    This is more flexible than the simple form.
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    ratio = k_safe / k_c
    numerator = ratio**n
    denominator = 1 + ratio**m
    
    T_norm = 1 + A * numerator / denominator
    
    return T_norm

def fit_function_to_data(k_data, T_data, function_name='simple', initial_guess=None):
    """
    Fit the specified function to data.
    
    Parameters:
    -----------
    k_data : array_like
        Wavenumber data
    T_data : array_like
        Transfer function data
    function_name : str
        'simple', 'exponential', or 'power_law'
    initial_guess : array_like, optional
        Initial parameter guess
        
    Returns:
    --------
    popt : array
        Optimal parameters
    pcov : array
        Parameter covariance matrix
    r2 : float
        Coefficient of determination
    """
    
    if function_name == 'simple':
        func = simple_idm_transfer_function
        if initial_guess is None:
            initial_guess = [0.1, 1.0, 2.0]  # A, k_c, n
        bounds = ([-1, 0.01, 0.1], [1, 100, 10])
    elif function_name == 'exponential':
        func = exponential_suppression_function
        if initial_guess is None:
            initial_guess = [0.1, 1.0, 2.0]  # A, k_c, alpha
        bounds = ([-1, 0.01, 0.1], [1, 100, 10])
    elif function_name == 'power_law':
        func = power_law_suppression_function
        if initial_guess is None:
            initial_guess = [0.1, 1.0, 2.0, 2.0]  # A, k_c, n, m
        bounds = ([-1, 0.01, 0.1, 0.1], [1, 100, 10, 10])
    else:
        raise ValueError("function_name must be 'simple', 'exponential', or 'power_law'")
    
    try:
        popt, pcov = curve_fit(func, k_data, T_data, p0=initial_guess, 
                              bounds=bounds, maxfev=5000)
        
        # Calculate R²
        T_pred = func(k_data, *popt)
        ss_res = np.sum((T_data - T_pred)**2)
        ss_tot = np.sum((T_data - np.mean(T_data))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return popt, pcov, r2
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, 0

def test_all_functions_on_data():
    """Test all function forms on sample data."""
    
    # Load LCDM data
    print("Loading LCDM transfer function...")
    try:
        lcdm_data = np.loadtxt('output/pk-lcdm_tk.dat', skiprows=9)
        k_lcdm = lcdm_data[:, 0]
        d_tot_lcdm = lcdm_data[:, 5]  # LCDM has 8 columns
    except Exception as e:
        print(f"Failed to load LCDM data: {e}")
        return
    
    # Find sample IDM files
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    print(f"Found {len(tk_files)} IDM files")
    
    # Test on a few files
    test_files = tk_files[:3]
    
    results = {}
    
    for filename in test_files:
        print(f"\nTesting file: {os.path.basename(filename)}")
        
        try:
            # Load IDM data
            idm_data = np.loadtxt(filename, skiprows=9)
            k_idm = idm_data[:, 0]
            d_tot_idm = idm_data[:, 6]  # IDM files have 9 columns
            
            # Normalize by LCDM
            d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
            T_normalized = d_tot_idm / d_tot_lcdm_interp
            
            # Filter for k > 0.01
            mask = k_idm > 0.01
            k_filtered = k_idm[mask]
            T_filtered = T_normalized[mask]
            
            if len(k_filtered) < 10:
                print("  Not enough data points after filtering")
                continue
            
            file_results = {}
            
            # Test simple function
            popt, pcov, r2 = fit_function_to_data(k_filtered, T_filtered, 'simple')
            if popt is not None:
                print(f"  Simple function R² = {r2:.4f}")
                print(f"    Parameters: A={popt[0]:.3f}, k_c={popt[1]:.3f}, n={popt[2]:.3f}")
                file_results['simple'] = {'params': popt, 'r2': r2, 'k_data': k_filtered, 'T_data': T_filtered}
            
            # Test exponential function
            popt, pcov, r2 = fit_function_to_data(k_filtered, T_filtered, 'exponential')
            if popt is not None:
                print(f"  Exponential function R² = {r2:.4f}")
                print(f"    Parameters: A={popt[0]:.3f}, k_c={popt[1]:.3f}, α={popt[2]:.3f}")
                file_results['exponential'] = {'params': popt, 'r2': r2, 'k_data': k_filtered, 'T_data': T_filtered}
            
            # Test power law function
            popt, pcov, r2 = fit_function_to_data(k_filtered, T_filtered, 'power_law')
            if popt is not None:
                print(f"  Power law function R² = {r2:.4f}")
                print(f"    Parameters: A={popt[0]:.3f}, k_c={popt[1]:.3f}, n={popt[2]:.3f}, m={popt[3]:.3f}")
                file_results['power_law'] = {'params': popt, 'r2': r2, 'k_data': k_filtered, 'T_data': T_filtered}
            
            results[os.path.basename(filename)] = file_results
            
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    return results

def plot_best_fit(results):
    """Plot the best fit from the results."""
    
    best_r2 = -np.inf
    best_file = None
    best_function = None
    best_data = None
    
    for filename, file_results in results.items():
        for func_name, func_data in file_results.items():
            if func_data['r2'] > best_r2:
                best_r2 = func_data['r2']
                best_file = filename
                best_function = func_name
                best_data = func_data
    
    if best_data is None:
        print("No successful fits found")
        return
    
    print(f"\nBest fit: {best_file} with {best_function} function (R² = {best_r2:.4f})")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    k_data = best_data['k_data']
    T_data = best_data['T_data']
    params = best_data['params']
    
    # Plot data
    plt.semilogx(k_data, T_data, 'o', alpha=0.6, markersize=4, label='Data')
    
    # Generate smooth curve
    k_smooth = np.logspace(np.log10(k_data.min()), np.log10(k_data.max()), 200)
    
    if best_function == 'simple':
        T_smooth = simple_idm_transfer_function(k_smooth, *params)
    elif best_function == 'exponential':
        T_smooth = exponential_suppression_function(k_smooth, *params)
    elif best_function == 'power_law':
        T_smooth = power_law_suppression_function(k_smooth, *params)
    
    plt.semilogx(k_smooth, T_smooth, 'r-', linewidth=2, label=f'{best_function.title()} Fit')
    
    # Add reference line
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title(f'Best Analytic Fit: {best_file}\n{best_function.title()} Function (R² = {best_r2:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('best_analytic_fit.png', dpi=300, bbox_inches='tight')
    print("Plot saved as best_analytic_fit.png")
    plt.show()

if __name__ == "__main__":
    print("Testing Simplified Analytic Functions")
    print("=" * 50)
    
    results = test_all_functions_on_data()
    
    if results:
        plot_best_fit(results)
        
        # Summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        all_r2 = []
        for filename, file_results in results.items():
            for func_name, func_data in file_results.items():
                all_r2.append(func_data['r2'])
                print(f"{filename} - {func_name}: R² = {func_data['r2']:.4f}")
        
        if all_r2:
            print(f"\nAverage R²: {np.mean(all_r2):.4f}")
            print(f"Best R²: {np.max(all_r2):.4f}")
    else:
        print("No successful fits obtained")

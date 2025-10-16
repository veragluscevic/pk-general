#!/usr/bin/env python3
"""
Final optimized analytic transfer function for interacting dark matter.

Based on testing results, this provides the best functional form
for describing IDM transfer functions with minimal parameters.

The recommended form is:
$$\frac{T_{\mathrm{IDM}}(k)}{T_{\Lambda\mathrm{CDM}}(k)} = 1 + A \frac{(k/k_c)^n}{1 + (k/k_c)^m}$$

This captures the essential physics with just 4 parameters:
- A: Amplitude of deviation
- k_c: Characteristic transition scale  
- n: Low-k power law index
- m: High-k suppression index
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

def idm_transfer_function(k, A, k_c, n, m):
    """
    Optimized 4-parameter analytic form for IDM transfer functions.
    
    This is the recommended form based on testing results.
    
    Parameters:
    -----------
    k : array_like
        Wavenumber in h/Mpc
    A : float
        Amplitude of deviation from ΛCDM
    k_c : float
        Characteristic transition scale in h/Mpc
    n : float
        Low-k power law index
    m : float
        High-k suppression index
        
    Returns:
    --------
    T_norm : array_like
        Normalized transfer function T_IDM(k) / T_ΛCDM(k)
        
    Notes:
    ------
    The function is: T_norm(k) = 1 + A * (k/k_c)^n / (1 + (k/k_c)^m)
    
    Physical interpretation:
    - For k << k_c: T_norm ≈ 1 + A * (k/k_c)^n
    - For k >> k_c: T_norm ≈ 1 + A * (k/k_c)^(n-m)
    - Transition occurs around k ≈ k_c
    - A > 0: enhancement, A < 0: suppression
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    ratio = k_safe / k_c
    numerator = ratio**n
    denominator = 1 + ratio**m
    
    T_norm = 1 + A * numerator / denominator
    
    return T_norm

def fit_idm_function(k_data, T_data, initial_guess=None):
    """
    Fit the IDM transfer function to data.
    
    Parameters:
    -----------
    k_data : array_like
        Wavenumber data
    T_data : array_like
        Transfer function data
    initial_guess : array_like, optional
        Initial parameter guess [A, k_c, n, m]
        
    Returns:
    --------
    popt : array
        Optimal parameters [A, k_c, n, m]
    pcov : array
        Parameter covariance matrix
    r2 : float
        Coefficient of determination
    """
    
    if initial_guess is None:
        initial_guess = [0.1, 1.0, 2.0, 2.0]
    
    # Set reasonable bounds
    bounds = ([-1, 0.01, 0.1, 0.1], [1, 100, 10, 10])
    
    try:
        popt, pcov = curve_fit(idm_transfer_function, k_data, T_data, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        # Calculate R²
        T_pred = idm_transfer_function(k_data, *popt)
        ss_res = np.sum((T_data - T_pred)**2)
        ss_tot = np.sum((T_data - np.mean(T_data))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return popt, pcov, r2
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, 0

def analyze_all_transfer_functions():
    """
    Analyze all transfer functions and fit the analytic form.
    
    Returns:
    --------
    results : dict
        Dictionary containing fit results for all files
    """
    
    # Load LCDM data
    print("Loading LCDM transfer function...")
    try:
        lcdm_data = np.loadtxt('output/pk-lcdm_tk.dat', skiprows=9)
        k_lcdm = lcdm_data[:, 0]
        d_tot_lcdm = lcdm_data[:, 5]
    except Exception as e:
        print(f"Failed to load LCDM data: {e}")
        return {}
    
    # Find all IDM files
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    print(f"Found {len(tk_files)} IDM files")
    
    results = {}
    successful_fits = 0
    
    for i, filename in enumerate(tk_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(tk_files)}...")
        
        try:
            # Load IDM data
            idm_data = np.loadtxt(filename, skiprows=9)
            k_idm = idm_data[:, 0]
            d_tot_idm = idm_data[:, 6]
            
            # Normalize by LCDM
            d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
            T_normalized = d_tot_idm / d_tot_lcdm_interp
            
            # Filter for k > 0.01
            mask = k_idm > 0.01
            k_filtered = k_idm[mask]
            T_filtered = T_normalized[mask]
            
            if len(k_filtered) < 10:
                continue
            
            # Fit the analytic function
            popt, pcov, r2 = fit_idm_function(k_filtered, T_filtered)
            
            if popt is not None and r2 > 0.5:  # Only keep good fits
                results[os.path.basename(filename)] = {
                    'params': popt,
                    'r2': r2,
                    'k_data': k_filtered,
                    'T_data': T_filtered
                }
                successful_fits += 1
        
        except Exception as e:
            continue
    
    print(f"Successfully fitted {successful_fits} transfer functions")
    return results

def create_parameter_summary(results):
    """Create a summary of the fitted parameters."""
    
    if not results:
        print("No results to summarize")
        return
    
    # Extract parameters
    A_values = [r['params'][0] for r in results.values()]
    k_c_values = [r['params'][1] for r in results.values()]
    n_values = [r['params'][2] for r in results.values()]
    m_values = [r['params'][3] for r in results.values()]
    r2_values = [r['r2'] for r in results.values()]
    
    print("\n" + "="*60)
    print("PARAMETER SUMMARY")
    print("="*60)
    print(f"Number of successful fits: {len(results)}")
    print(f"Average R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
    print(f"Best R²: {np.max(r2_values):.4f}")
    print(f"Worst R²: {np.min(r2_values):.4f}")
    
    print(f"\nAmplitude (A):")
    print(f"  Mean: {np.mean(A_values):.4f} ± {np.std(A_values):.4f}")
    print(f"  Range: [{np.min(A_values):.4f}, {np.max(A_values):.4f}]")
    
    print(f"\nTransition scale (k_c):")
    print(f"  Mean: {np.mean(k_c_values):.4f} ± {np.std(k_c_values):.4f}")
    print(f"  Range: [{np.min(k_c_values):.4f}, {np.max(k_c_values):.4f}]")
    
    print(f"\nLow-k power (n):")
    print(f"  Mean: {np.mean(n_values):.4f} ± {np.std(n_values):.4f}")
    print(f"  Range: [{np.min(n_values):.4f}, {np.max(n_values):.4f}]")
    
    print(f"\nHigh-k suppression (m):")
    print(f"  Mean: {np.mean(m_values):.4f} ± {np.std(m_values):.4f}")
    print(f"  Range: [{np.min(m_values):.4f}, {np.max(m_values):.4f}]")

def plot_representative_fits(results, n_plots=3):
    """Plot representative fits."""
    
    if not results:
        print("No results to plot")
        return
    
    # Sort by R² and take the best fits
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    best_results = sorted_results[:n_plots]
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    for i, (filename, data) in enumerate(best_results):
        ax = axes[i]
        
        k_data = data['k_data']
        T_data = data['T_data']
        params = data['params']
        r2 = data['r2']
        
        # Plot data
        ax.semilogx(k_data, T_data, 'o', alpha=0.6, markersize=3, label='Data')
        
        # Plot fit
        k_smooth = np.logspace(np.log10(k_data.min()), np.log10(k_data.max()), 200)
        T_smooth = idm_transfer_function(k_smooth, *params)
        ax.semilogx(k_smooth, T_smooth, 'r-', linewidth=2, label='Fit')
        
        # Reference line
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
        
        ax.set_xlabel('k [h/Mpc]')
        ax.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
        ax.set_title(f'{filename}\nR² = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('representative_analytic_fits.png', dpi=300, bbox_inches='tight')
    print("Representative fits saved as representative_analytic_fits.png")
    plt.show()

if __name__ == "__main__":
    print("Final Analytic Transfer Function Analysis")
    print("=" * 50)
    
    # Analyze all transfer functions
    results = analyze_all_transfer_functions()
    
    if results:
        # Create parameter summary
        create_parameter_summary(results)
        
        # Plot representative fits
        plot_representative_fits(results)
        
        print(f"\nAnalysis complete! Successfully fitted {len(results)} transfer functions.")
        print("The recommended analytic form is:")
        print("T_IDM(k) / T_ΛCDM(k) = 1 + A * (k/k_c)^n / (1 + (k/k_c)^m)")
        print("\nWith parameters:")
        print("- A: Amplitude of deviation")
        print("- k_c: Characteristic transition scale")
        print("- n: Low-k power law index") 
        print("- m: High-k suppression index")
    else:
        print("No successful fits obtained")

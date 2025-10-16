#!/usr/bin/env python3
"""
Fit the enhanced oscillatory function to real transfer function data.

This script loads actual transfer function data from the output folder,
fits the enhanced function, and plots the comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

def enhanced_oscillatory_function(k, A, k_c, alpha, A0, beta, k_decay, k0, gamma, phi0, delta, k_ref=1.0, phi1=0.0):
    """
    Enhanced oscillatory transfer function with k-dependent parameters.
    """
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Base exponential function
    T_base = 1 + A * np.exp(-(k_safe / k_c)**alpha)
    
    # k-dependent oscillation amplitude
    A_osc_k = A0 * (k_safe / k_ref)**beta * np.exp(-k_safe / k_decay)
    
    # k-dependent oscillation frequency
    k_osc_k = k0 * (k_safe / k_ref)**gamma
    
    # k-dependent phase
    phi_k = phi0 + phi1 * (k_safe / k_ref)**delta
    
    # Oscillatory component
    oscillation = A_osc_k * np.cos(2 * np.pi * k_safe / k_osc_k + phi_k)
    
    T_norm = T_base + oscillation
    
    return T_norm

def simplified_oscillatory_function(k, A, k_c, alpha, A0, k_decay, k0, phi0):
    """
    Simplified version with fewer parameters for initial fitting.
    """
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Base exponential function
    T_base = 1 + A * np.exp(-(k_safe / k_c)**alpha)
    
    # Simple oscillatory component
    A_osc_k = A0 * np.exp(-k_safe / k_decay)
    oscillation = A_osc_k * np.cos(2 * np.pi * k_safe / k0 + phi0)
    
    T_norm = T_base + oscillation
    
    return T_norm

def load_transfer_function_data(filename):
    """Load transfer function data from CLASS output."""
    try:
        data = np.loadtxt(filename, skiprows=9)
        k = data[:, 0]
        
        # Handle different column structures
        if data.shape[1] == 9:
            d_tot = data[:, 6]  # IDM files
        elif data.shape[1] == 8:
            d_tot = data[:, 5]  # LCDM files
        else:
            raise ValueError(f"Unexpected number of columns: {data.shape[1]}")
        
        return k, d_tot
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def load_lcdm_data():
    """Load LCDM data for normalization."""
    lcdm_file = 'output/pk-lcdm_tk.dat'
    k, d_tot = load_transfer_function_data(lcdm_file)
    return k, d_tot

def fit_function_to_data(k_data, T_data, function_type='simplified'):
    """
    Fit the oscillatory function to data.
    
    Parameters:
    -----------
    k_data : array_like
        Wavenumber data
    T_data : array_like
        Transfer function data
    function_type : str
        'simplified' or 'enhanced'
        
    Returns:
    --------
    popt : array
        Optimal parameters
    pcov : array
        Parameter covariance matrix
    r2 : float
        Coefficient of determination
    """
    
    if function_type == 'simplified':
        func = simplified_oscillatory_function
        # Initial guess: [A, k_c, alpha, A0, k_decay, k0, phi0]
        initial_guess = [0.1, 1.0, 2.0, 0.05, 2.0, 0.5, 0.0]
        bounds = ([-1, 0.01, 0.1, 0, 0.1, 0.01, -np.pi], [1, 100, 10, 1, 100, 10, np.pi])
    elif function_type == 'enhanced':
        func = enhanced_oscillatory_function
        # Initial guess: [A, k_c, alpha, A0, beta, k_decay, k0, gamma, phi0, delta]
        initial_guess = [0.1, 1.0, 2.0, 0.05, 0.0, 2.0, 0.5, 0.0, 0.0, 0.0]
        bounds = ([-1, 0.01, 0.1, 0, -2, 0.1, 0.01, -2, -np.pi, -2], 
                 [1, 100, 10, 1, 2, 100, 10, 2, np.pi, 2])
    else:
        raise ValueError("function_type must be 'simplified' or 'enhanced'")
    
    try:
        popt, pcov = curve_fit(func, k_data, T_data, p0=initial_guess, 
                              bounds=bounds, maxfev=10000)
        
        # Calculate R²
        T_pred = func(k_data, *popt)
        ss_res = np.sum((T_data - T_pred)**2)
        ss_tot = np.sum((T_data - np.mean(T_data))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return popt, pcov, r2
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, 0

def fit_and_plot_real_data():
    """Load real data, fit the function, and plot comparison."""
    
    print("Loading LCDM data for normalization...")
    k_lcdm, d_tot_lcdm = load_lcdm_data()
    if k_lcdm is None:
        print("Failed to load LCDM data")
        return
    
    # Find a sample IDM file to fit
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    if not tk_files:
        print("No IDM files found")
        return
    
    # Pick a representative file
    sample_file = tk_files[10]  # Pick the 11th file as a sample
    print(f"Fitting file: {os.path.basename(sample_file)}")
    
    # Load IDM data
    k_idm, d_tot_idm = load_transfer_function_data(sample_file)
    if k_idm is None:
        print("Failed to load IDM data")
        return
    
    # Normalize by LCDM
    d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
    T_normalized = d_tot_idm / d_tot_lcdm_interp
    
    # Filter for k > 0.01
    mask = k_idm > 0.01
    k_filtered = k_idm[mask]
    T_filtered = T_normalized[mask]
    
    print(f"Data points after filtering: {len(k_filtered)}")
    
    if len(k_filtered) < 20:
        print("Not enough data points for fitting")
        return
    
    # Fit simplified function first
    print("Fitting simplified function...")
    popt_simple, pcov_simple, r2_simple = fit_function_to_data(k_filtered, T_filtered, 'simplified')
    
    if popt_simple is not None:
        print(f"Simplified fit R² = {r2_simple:.4f}")
        print(f"Parameters: A={popt_simple[0]:.3f}, k_c={popt_simple[1]:.3f}, α={popt_simple[2]:.3f}")
        print(f"          A0={popt_simple[3]:.3f}, k_decay={popt_simple[4]:.3f}, k0={popt_simple[5]:.3f}, φ0={popt_simple[6]:.3f}")
    else:
        print("Simplified fit failed")
        return
    
    # Try enhanced function if simplified fit is good
    if r2_simple > 0.5:
        print("Trying enhanced function...")
        popt_enhanced, pcov_enhanced, r2_enhanced = fit_function_to_data(k_filtered, T_filtered, 'enhanced')
        
        if popt_enhanced is not None:
            print(f"Enhanced fit R² = {r2_enhanced:.4f}")
            use_enhanced = r2_enhanced > r2_simple
        else:
            print("Enhanced fit failed, using simplified")
            use_enhanced = False
    else:
        print("Simplified fit not good enough, skipping enhanced")
        use_enhanced = False
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot data
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    
    # Generate smooth curve for plotting
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    
    # Plot fitted function
    if use_enhanced:
        T_smooth = enhanced_oscillatory_function(k_smooth, *popt_enhanced)
        r2_plot = r2_enhanced
        fit_label = 'Enhanced Fit'
        color = 'red'
        linewidth = 2.5
    else:
        T_smooth = simplified_oscillatory_function(k_smooth, *popt_simple)
        r2_plot = r2_simple
        fit_label = 'Simplified Fit'
        color = 'blue'
        linewidth = 2
    
    plt.semilogx(k_smooth, T_smooth, color=color, linewidth=linewidth, 
                label=f'{fit_label} (R² = {r2_plot:.4f})')
    
    # Add reference line
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title(f'Real Transfer Function Fit\n{os.path.basename(sample_file)}', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    if use_enhanced:
        param_text = f'Enhanced Parameters:\nA={popt_enhanced[0]:.3f}, k_c={popt_enhanced[1]:.3f}, α={popt_enhanced[2]:.3f}\nA0={popt_enhanced[3]:.3f}, β={popt_enhanced[4]:.3f}, k_decay={popt_enhanced[5]:.3f}\nk0={popt_enhanced[6]:.3f}, γ={popt_enhanced[7]:.3f}, φ0={popt_enhanced[8]:.3f}, δ={popt_enhanced[9]:.3f}'
    else:
        param_text = f'Simplified Parameters:\nA={popt_simple[0]:.3f}, k_c={popt_simple[1]:.3f}, α={popt_simple[2]:.3f}\nA0={popt_simple[3]:.3f}, k_decay={popt_simple[4]:.3f}\nk0={popt_simple[5]:.3f}, φ0={popt_simple[6]:.3f}'
    
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=9, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('real_transfer_function_fit.png', dpi=300, bbox_inches='tight')
    print("Real transfer function fit saved as real_transfer_function_fit.png")
    plt.show()
    
    return popt_simple if not use_enhanced else popt_enhanced, r2_plot

if __name__ == "__main__":
    print("Fitting enhanced oscillatory function to real transfer function data...")
    print("=" * 70)
    
    # Fit and plot
    result = fit_and_plot_real_data()
    
    if result is not None:
        params, r2 = result
        print(f"\nFitting completed successfully!")
        print(f"Best fit R² = {r2:.4f}")
        print("Parameters saved and plot generated.")
    else:
        print("\nFitting failed. Please check the data and try again.")

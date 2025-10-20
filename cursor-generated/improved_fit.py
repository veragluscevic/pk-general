#!/usr/bin/env python3
"""
Improved fitting of the enhanced oscillatory function to real transfer function data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

def simplified_oscillatory_function(k, A, k_c, alpha, A0, k_decay, k0, phi0):
    """
    Simplified oscillatory function for fitting.
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

def improved_fit():
    """Improved fitting with better initial guesses and bounds."""
    
    # Load LCDM data
    print("Loading LCDM data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Load IDM data
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    sample_file = tk_files[10]  # Same file as before
    print(f"Fitting file: {os.path.basename(sample_file)}")
    
    k_idm, d_tot_idm = load_transfer_function_data(sample_file)
    
    # Normalize by LCDM
    d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
    T_normalized = d_tot_idm / d_tot_lcdm_interp
    
    # Filter for k > 0.01
    mask = k_idm > 0.01
    k_filtered = k_idm[mask]
    T_filtered = T_normalized[mask]
    
    print(f"Data points: {len(k_filtered)}")
    print(f"T range: {T_filtered.min():.4f} to {T_filtered.max():.4f}")
    
    # Try multiple initial guesses
    initial_guesses = [
        # Guess 1: Based on data characteristics
        [0.1, 5.0, 2.0, 0.1, 3.0, 0.3, 0.0],
        # Guess 2: Conservative
        [0.05, 2.0, 1.5, 0.05, 2.0, 0.5, 0.0],
        # Guess 3: More oscillatory
        [0.2, 3.0, 2.5, 0.15, 1.5, 0.4, 1.0],
        # Guess 4: High k_c
        [0.1, 10.0, 2.0, 0.08, 5.0, 0.2, 0.0],
        # Guess 5: Low amplitude
        [0.02, 1.5, 1.0, 0.03, 1.0, 0.6, 0.5],
    ]
    
    # Improved bounds - constrain k_c to 1-20 as suggested
    bounds = (
        [-1, 1.0, 0.1, 0, 0.1, 0.01, -np.pi],  # Lower bounds
        [1, 20.0, 5.0, 1, 20, 5, np.pi]        # Upper bounds
    )
    
    best_fit = None
    best_r2 = -np.inf
    best_params = None
    
    for i, initial_guess in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}: {initial_guess}")
        
        try:
            popt, pcov = curve_fit(simplified_oscillatory_function, k_filtered, T_filtered, 
                                  p0=initial_guess, bounds=bounds, maxfev=10000)
            
            # Calculate R²
            T_pred = simplified_oscillatory_function(k_filtered, *popt)
            ss_res = np.sum((T_filtered - T_pred)**2)
            ss_tot = np.sum((T_filtered - np.mean(T_filtered))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"  R² = {r2:.4f}")
            print(f"  Parameters: A={popt[0]:.3f}, k_c={popt[1]:.3f}, α={popt[2]:.3f}")
            print(f"             A0={popt[3]:.3f}, k_decay={popt[4]:.3f}, k0={popt[5]:.3f}, φ0={popt[6]:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = popt
                best_fit = (popt, pcov, r2)
                
        except Exception as e:
            print(f"  Fit failed: {e}")
    
    if best_fit is None:
        print("All fits failed!")
        return
    
    print(f"\nBest fit R² = {best_r2:.4f}")
    
    # Plot the best fit
    plt.figure(figsize=(12, 8))
    
    # Plot data
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    
    # Generate smooth curve for plotting
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    T_smooth = simplified_oscillatory_function(k_smooth, *best_params)
    plt.semilogx(k_smooth, T_smooth, 'r-', linewidth=2.5, 
                label=f'Best Fit (R² = {best_r2:.4f})')
    
    # Add reference line
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title(f'Improved Fit: {os.path.basename(sample_file)}', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'Best Fit Parameters:\nA={best_params[0]:.3f}, k_c={best_params[1]:.3f}, α={best_params[2]:.3f}\nA0={best_params[3]:.3f}, k_decay={best_params[4]:.3f}\nk0={best_params[5]:.3f}, φ0={best_params[6]:.3f}'
    
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('improved_fit.png', dpi=300, bbox_inches='tight')
    print("Improved fit saved as improved_fit.png")
    plt.show()
    
    return best_params, best_r2

if __name__ == "__main__":
    print("Improved fitting of transfer function...")
    print("=" * 50)
    
    result = improved_fit()
    
    if result is not None:
        params, r2 = result
        print(f"\nFitting completed successfully!")
        print(f"Best R² = {r2:.4f}")
        print("This should be much better than the previous R² = -0.17")

#!/usr/bin/env python3
"""
Fit the correct np0 file that actually has minimal oscillations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

def dao_transfer_function(k, alpha, beta, gamma, rs, kD, A, phi, m):
    """DAO transfer function model."""
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Envelope function: T_env(k) = [1 + (αk)^β]^γ
    T_env = (1 + (alpha * k_safe)**beta)**gamma
    
    # DAO function: T_DAO(k) = 1 + A sin(k r_s + φ) exp[-(k/k_D)^m]
    T_DAO = 1 + A * np.sin(k_safe * rs + phi) * np.exp(-(k_safe / kD)**m)
    
    T_IDM = T_env * T_DAO
    
    return T_IDM

def envelope_only_function(k, alpha, beta, gamma):
    """Envelope-only function for np0 files."""
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Envelope function: T_env(k) = [1 + (αk)^β]^γ
    T_env = (1 + (alpha * k_safe)**beta)**gamma
    
    return T_env

def load_transfer_function_data(filename):
    """Load transfer function data from CLASS output."""
    try:
        data = np.loadtxt(filename, skiprows=9)
        k = data[:, 0]
        
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

def fit_correct_np0():
    """Fit the correct np0 file that actually has minimal oscillations."""
    
    # Load data
    print("Loading data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Find np0 files and pick one with minimal oscillations
    tk_files = glob.glob('output/*tk.dat')
    np0_files = [f for f in tk_files if 'np0' in f and 'lcdm' not in f.lower()]
    
    # Test a few np0 files to find one with minimal oscillations
    best_np0_file = None
    min_std = float('inf')
    
    for np0_file in np0_files[:10]:  # Test first 10 files
        k_idm, d_tot_idm = load_transfer_function_data(np0_file)
        d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
        T_normalized = d_tot_idm / d_tot_lcdm_interp
        
        mask = k_idm > 0.01
        T_filtered = T_normalized[mask]
        T_std = np.std(T_filtered)
        
        print(f"Testing {os.path.basename(np0_file)}: std = {T_std:.4f}")
        
        if T_std < min_std:
            min_std = T_std
            best_np0_file = np0_file
    
    if best_np0_file is None:
        print("No suitable np0 file found!")
        return
    
    print(f"\nBest np0 file (minimal oscillations): {os.path.basename(best_np0_file)}")
    print(f"Standard deviation: {min_std:.4f}")
    
    # Load the best np0 file
    k_idm, d_tot_idm = load_transfer_function_data(best_np0_file)
    
    # Normalize by LCDM
    d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
    T_normalized = d_tot_idm / d_tot_lcdm_interp
    
    # Filter for k > 0.01
    mask = k_idm > 0.01
    k_filtered = k_idm[mask]
    T_filtered = T_normalized[mask]
    
    print(f"Data points: {len(k_filtered)}")
    print(f"T range: {T_filtered.min():.4f} to {T_filtered.max():.4f}")
    print(f"T std: {np.std(T_filtered):.4f}")
    
    # Try envelope-only fit first (3 parameters)
    print("\n=== ENVELOPE-ONLY FIT (3 parameters) ===")
    
    initial_guesses_envelope = [
        [0.1, 1.0, -0.1],
        [0.2, 1.5, -0.2],
        [0.15, 0.8, -0.05],
        [0.05, 0.5, -0.01],
        [0.3, 2.0, -0.3],
    ]
    
    bounds_envelope = (
        [0.001, 0.1, -2.0],  # Lower bounds
        [1.0, 5.0, -0.001]   # Upper bounds
    )
    
    best_envelope_fit = None
    best_envelope_r2 = -np.inf
    best_envelope_params = None
    
    for i, initial_guess in enumerate(initial_guesses_envelope):
        print(f"Trying envelope guess {i+1}: {initial_guess}")
        
        try:
            popt, pcov = curve_fit(envelope_only_function, k_filtered, T_filtered, 
                                  p0=initial_guess, bounds=bounds_envelope, maxfev=20000)
            
            # Calculate R²
            T_pred = envelope_only_function(k_filtered, *popt)
            ss_res = np.sum((T_filtered - T_pred)**2)
            ss_tot = np.sum((T_filtered - np.mean(T_filtered))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"  R² = {r2:.4f}")
            print(f"  Parameters: α={popt[0]:.3f}, β={popt[1]:.3f}, γ={popt[2]:.3f}")
            
            if r2 > best_envelope_r2:
                best_envelope_r2 = r2
                best_envelope_params = popt
                best_envelope_fit = (popt, pcov, r2)
                
        except Exception as e:
            print(f"  Fit failed: {e}")
    
    if best_envelope_fit is None:
        print("All envelope fits failed!")
        return
    
    print(f"\nBest envelope-only fit R² = {best_envelope_r2:.4f}")
    
    # Plot the results
    plt.figure(figsize=(16, 10))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.8, markersize=5, 
                label='Data (should be smooth)', color='black')
    
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    
    # Envelope-only fit
    T_envelope = envelope_only_function(k_smooth, *best_envelope_params)
    plt.semilogx(k_smooth, T_envelope, 'b-', linewidth=3, 
                label=f'Envelope Only (R² = {best_envelope_r2:.4f})')
    
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM reference')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title(f'Correct np0 File: {os.path.basename(best_np0_file)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed view
    plt.subplot(2, 2, 2)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.8, markersize=5, 
                label='Data', color='black')
    plt.semilogx(k_smooth, T_envelope, 'b-', linewidth=3, 
                label=f'Envelope Fit (R² = {best_envelope_r2:.4f})')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM reference')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title('Zoomed View - Should Show Smooth Behavior')
    plt.ylim(0.95, 1.05)  # Zoom in on the relevant range
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Envelope function
    plt.subplot(2, 2, 3)
    alpha, beta, gamma = best_envelope_params
    T_env = (1 + (alpha * k_smooth)**beta)**gamma
    plt.semilogx(k_smooth, T_env, 'b-', linewidth=2, label='T_env(k)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No effect')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_env(k)')
    plt.title(f'Envelope: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(2, 2, 4)
    T_pred = envelope_only_function(k_filtered, *best_envelope_params)
    residuals = T_filtered - T_pred
    plt.semilogx(k_filtered, residuals, 'o', alpha=0.8, markersize=4, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Residuals')
    plt.title(f'Residuals (RMS = {np.sqrt(np.mean(residuals**2)):.6f})')
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'Correct np0 File Analysis:\nFile: {os.path.basename(best_np0_file)}\nT range: {T_filtered.min():.4f} to {T_filtered.max():.4f}\nT std: {np.std(T_filtered):.4f}\n\nEnvelope: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}'
    
    plt.figtext(0.02, 0.02, param_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('correct_np0_fit.png', dpi=300, bbox_inches='tight')
    print("Correct np0 fit saved as correct_np0_fit.png")
    plt.show()
    
    return best_envelope_params, best_envelope_r2

if __name__ == "__main__":
    print("Fitting the correct np0 file with minimal oscillations...")
    print("=" * 60)
    
    result = fit_correct_np0()
    
    if result is not None:
        env_params, env_r2 = result
        print(f"\nCorrect np0 file fitting completed!")
        print(f"Envelope-only R² = {env_r2:.4f}")
        print("This should show smooth data with minimal oscillations!")

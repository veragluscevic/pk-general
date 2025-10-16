#!/usr/bin/env python3
"""
Fit a np0 file (minimal oscillations) to test the envelope function.
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

def fit_np0_file():
    """Fit a np0 file to test envelope function without oscillations."""
    
    # Load data
    print("Loading data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Find np0 files
    tk_files = glob.glob('output/*tk.dat')
    np0_files = [f for f in tk_files if 'np0' in f and 'lcdm' not in f.lower()]
    
    if not np0_files:
        print("No np0 files found!")
        return
    
    # Use the first np0 file
    sample_file = np0_files[0]
    print(f"Fitting np0 file: {os.path.basename(sample_file)}")
    
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
    
    # Try envelope-only fit first (3 parameters)
    print("\n=== ENVELOPE-ONLY FIT (3 parameters) ===")
    
    initial_guesses_envelope = [
        [0.1, 1.0, -0.5],
        [0.2, 1.5, -0.3],
        [0.15, 0.8, -0.4],
        [0.25, 2.0, -0.6],
        [0.05, 0.5, -0.2],
    ]
    
    bounds_envelope = (
        [0.001, 0.1, -3.0],  # Lower bounds
        [1.0, 5.0, -0.01]    # Upper bounds
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
    
    # Now try full DAO fit (8 parameters)
    print("\n=== FULL DAO FIT (8 parameters) ===")
    
    initial_guesses_dao = [
        # Start with envelope params + small DAO
        [best_envelope_params[0], best_envelope_params[1], best_envelope_params[2], 
         1.0, 2.0, 0.1, 0.0, 2.0],
        # Try with larger DAO
        [best_envelope_params[0], best_envelope_params[1], best_envelope_params[2], 
         0.8, 1.5, 0.2, 0.0, 1.5],
        # Different scales
        [best_envelope_params[0], best_envelope_params[1], best_envelope_params[2], 
         1.2, 2.5, 0.05, 0.5, 2.5],
    ]
    
    bounds_dao = (
        [0.001, 0.1, -3.0, 0.01, 0.01, 0.0, -np.pi, 0.01],  # Lower bounds
        [1.0, 5.0, -0.01, 20.0, 20.0, 1.0, np.pi, 10.0]    # Upper bounds
    )
    
    best_dao_fit = None
    best_dao_r2 = -np.inf
    best_dao_params = None
    
    for i, initial_guess in enumerate(initial_guesses_dao):
        print(f"Trying DAO guess {i+1}: {initial_guess}")
        
        try:
            popt, pcov = curve_fit(dao_transfer_function, k_filtered, T_filtered, 
                                  p0=initial_guess, bounds=bounds_dao, maxfev=20000)
            
            # Calculate R²
            T_pred = dao_transfer_function(k_filtered, *popt)
            ss_res = np.sum((T_filtered - T_pred)**2)
            ss_tot = np.sum((T_filtered - np.mean(T_filtered))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"  R² = {r2:.4f}")
            print(f"  Parameters: α={popt[0]:.3f}, β={popt[1]:.3f}, γ={popt[2]:.3f}")
            print(f"             r_s={popt[3]:.3f}, k_D={popt[4]:.3f}, A={popt[5]:.3f}, φ={popt[6]:.3f}, m={popt[7]:.3f}")
            
            if r2 > best_dao_r2:
                best_dao_r2 = r2
                best_dao_params = popt
                best_dao_fit = (popt, pcov, r2)
                
        except Exception as e:
            print(f"  Fit failed: {e}")
    
    if best_dao_fit is None:
        print("All DAO fits failed!")
        return
    
    print(f"\nBest DAO fit R² = {best_dao_r2:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(16, 10))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    
    # Envelope-only fit
    T_envelope = envelope_only_function(k_smooth, *best_envelope_params)
    plt.semilogx(k_smooth, T_envelope, 'b-', linewidth=2.5, 
                label=f'Envelope Only (R² = {best_envelope_r2:.4f})')
    
    # Full DAO fit
    T_dao = dao_transfer_function(k_smooth, *best_dao_params)
    plt.semilogx(k_smooth, T_dao, 'r-', linewidth=2.5, 
                label=f'Full DAO (R² = {best_dao_r2:.4f})')
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title(f'np0 File Fit: {os.path.basename(sample_file)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Envelope comparison
    plt.subplot(2, 2, 2)
    alpha_env, beta_env, gamma_env = best_envelope_params
    alpha_dao, beta_dao, gamma_dao = best_dao_params[0], best_dao_params[1], best_dao_params[2]
    
    T_env_env = (1 + (alpha_env * k_smooth)**beta_env)**gamma_env
    T_env_dao = (1 + (alpha_dao * k_smooth)**beta_dao)**gamma_dao
    
    plt.semilogx(k_smooth, T_env_env, 'b-', linewidth=2, label='Envelope Only')
    plt.semilogx(k_smooth, T_env_dao, 'r--', linewidth=2, label='From DAO Fit')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No effect')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_env(k)')
    plt.title('Envelope Function Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DAO component
    plt.subplot(2, 2, 3)
    rs, kD, A, phi, m = best_dao_params[3], best_dao_params[4], best_dao_params[5], best_dao_params[6], best_dao_params[7]
    T_DAO = 1 + A * np.sin(k_smooth * rs + phi) * np.exp(-(k_smooth / kD)**m)
    plt.semilogx(k_smooth, T_DAO, 'g-', linewidth=2, label='T_DAO(k)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No oscillations')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_DAO(k)')
    plt.title(f'DAO: r_s={rs:.3f}, k_D={kD:.3f}, A={A:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals comparison
    plt.subplot(2, 2, 4)
    T_pred_env = envelope_only_function(k_filtered, *best_envelope_params)
    T_pred_dao = dao_transfer_function(k_filtered, *best_dao_params)
    
    residuals_env = T_filtered - T_pred_env
    residuals_dao = T_filtered - T_pred_dao
    
    plt.semilogx(k_filtered, residuals_env, 'o', alpha=0.6, markersize=3, color='blue', label='Envelope Only')
    plt.semilogx(k_filtered, residuals_dao, 'o', alpha=0.6, markersize=3, color='red', label='Full DAO')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Residuals')
    plt.title(f'Residuals: Env RMS={np.sqrt(np.mean(residuals_env**2)):.4f}, DAO RMS={np.sqrt(np.mean(residuals_dao**2)):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'np0 File Analysis:\n\nEnvelope Only: α={alpha_env:.3f}, β={beta_env:.3f}, γ={gamma_env:.3f}\nFull DAO: α={alpha_dao:.3f}, β={beta_dao:.3f}, γ={gamma_dao:.3f}\nr_s={rs:.3f}, k_D={kD:.3f}, A={A:.3f}'
    
    plt.figtext(0.02, 0.02, param_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('np0_file_fit.png', dpi=300, bbox_inches='tight')
    print("np0 file fit saved as np0_file_fit.png")
    plt.show()
    
    return best_envelope_params, best_envelope_r2, best_dao_params, best_dao_r2

if __name__ == "__main__":
    print("Fitting np0 file (minimal oscillations)...")
    print("=" * 50)
    
    result = fit_np0_file()
    
    if result is not None:
        env_params, env_r2, dao_params, dao_r2 = result
        print(f"\nnp0 file fitting completed!")
        print(f"Envelope-only R² = {env_r2:.4f}")
        print(f"Full DAO R² = {dao_r2:.4f}")
        print("This should show if the envelope function works well without oscillations!")

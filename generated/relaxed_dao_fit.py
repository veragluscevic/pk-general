#!/usr/bin/env python3
"""
Relaxed DAO fitting with much more flexible bounds.
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

def relaxed_dao_fit():
    """Relaxed DAO fitting with much more flexible bounds."""
    
    # Load data
    print("Loading data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    sample_file = tk_files[10]
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
    
    # Much more relaxed initial guesses with larger DAO amplitude
    initial_guesses = [
        # Guess 1: Soft envelope, large DAO
        [0.1, 1.0, -0.3, 1.0, 2.0, 0.5, 0.0, 2.0],
        # Guess 2: Very soft envelope
        [0.05, 0.8, -0.2, 0.8, 1.5, 0.3, 0.0, 1.5],
        # Guess 3: Moderate soft parameters
        [0.15, 1.5, -0.4, 1.2, 2.5, 0.4, 0.5, 2.5],
        # Guess 4: Different scales with large A
        [0.08, 1.2, -0.25, 2.0, 3.0, 0.6, 1.0, 2.0],
        # Guess 5: Conservative but large A
        [0.12, 1.0, -0.15, 0.6, 1.0, 0.2, 0.0, 1.0],
        # Guess 6: Even softer
        [0.2, 0.5, -0.1, 1.5, 2.0, 0.4, 0.0, 1.5],
    ]
    
    # Much more relaxed bounds
    bounds = (
        [0.001, 0.1, -5.0, 0.01, 0.01, 0.0, -np.pi, 0.01],  # Lower bounds
        [2.0, 10.0, 2.0, 20.0, 20.0, 5.0, np.pi, 10.0]     # Upper bounds
    )
    
    print("\nMuch more relaxed bounds:")
    print("α: [0.001, 2.0] (envelope scale)")
    print("β: [0.1, 10.0] (envelope power - much more flexible)")
    print("γ: [-5.0, 2.0] (envelope power - much more flexible)")
    print("r_s: [0.01, 20.0] (sound horizon)")
    print("k_D: [0.01, 20.0] (damping scale)")
    print("A: [0.0, 5.0] (DAO amplitude - much larger)")
    print("φ: [-π, π] (phase)")
    print("m: [0.01, 10.0] (damping power)")
    
    best_fit = None
    best_r2 = -np.inf
    best_params = None
    
    for i, initial_guess in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}: {initial_guess}")
        
        try:
            popt, pcov = curve_fit(dao_transfer_function, k_filtered, T_filtered, 
                                  p0=initial_guess, bounds=bounds, maxfev=30000)
            
            # Calculate R²
            T_pred = dao_transfer_function(k_filtered, *popt)
            ss_res = np.sum((T_filtered - T_pred)**2)
            ss_tot = np.sum((T_filtered - np.mean(T_filtered))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"  R² = {r2:.4f}")
            print(f"  Parameters: α={popt[0]:.3f}, β={popt[1]:.3f}, γ={popt[2]:.3f}")
            print(f"             r_s={popt[3]:.3f}, k_D={popt[4]:.3f}, A={popt[5]:.3f}, φ={popt[6]:.3f}, m={popt[7]:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = popt
                best_fit = (popt, pcov, r2)
                
        except Exception as e:
            print(f"  Fit failed: {e}")
    
    if best_fit is None:
        print("All fits failed!")
        return
    
    print(f"\nBest relaxed DAO fit R² = {best_r2:.4f}")
    
    # Plot the best fit with components
    plt.figure(figsize=(15, 10))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    T_smooth = dao_transfer_function(k_smooth, *best_params)
    plt.semilogx(k_smooth, T_smooth, 'r-', linewidth=2.5, 
                label=f'Relaxed DAO Fit (R² = {best_r2:.4f})')
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title(f'Relaxed DAO Fit: {os.path.basename(sample_file)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Envelope function
    plt.subplot(2, 2, 2)
    alpha, beta, gamma = best_params[0], best_params[1], best_params[2]
    T_env = (1 + (alpha * k_smooth)**beta)**gamma
    plt.semilogx(k_smooth, T_env, 'b-', linewidth=2, label='T_env(k)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No effect')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_env(k)')
    plt.title(f'Envelope: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # DAO function
    plt.subplot(2, 2, 3)
    rs, kD, A, phi, m = best_params[3], best_params[4], best_params[5], best_params[6], best_params[7]
    T_DAO = 1 + A * np.sin(k_smooth * rs + phi) * np.exp(-(k_smooth / kD)**m)
    plt.semilogx(k_smooth, T_DAO, 'g-', linewidth=2, label='T_DAO(k)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No oscillations')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_DAO(k)')
    plt.title(f'DAO: r_s={rs:.3f}, k_D={kD:.3f}, A={A:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(2, 2, 4)
    T_pred = dao_transfer_function(k_filtered, *best_params)
    residuals = T_filtered - T_pred
    plt.semilogx(k_filtered, residuals, 'o', alpha=0.6, markersize=3, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Residuals')
    plt.title(f'Residuals (RMS = {np.sqrt(np.mean(residuals**2)):.4f})')
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'Relaxed DAO Parameters:\nα={best_params[0]:.3f}, β={best_params[1]:.3f}, γ={best_params[2]:.3f}\nr_s={best_params[3]:.3f}, k_D={best_params[4]:.3f}, A={best_params[5]:.3f}\nφ={best_params[6]:.3f}, m={best_params[7]:.3f}'
    
    plt.figtext(0.02, 0.02, param_text, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('relaxed_dao_fit.png', dpi=300, bbox_inches='tight')
    print("Relaxed DAO fit saved as relaxed_dao_fit.png")
    plt.show()
    
    return best_params, best_r2

if __name__ == "__main__":
    print("Relaxed DAO fitting with much more flexible bounds...")
    print("=" * 70)
    
    result = relaxed_dao_fit()
    
    if result is not None:
        params, r2 = result
        print(f"\nRelaxed DAO fitting completed!")
        print(f"Best R² = {r2:.4f}")
        print("This should finally have softer slopes and larger DAO amplitude!")

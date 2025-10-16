#!/usr/bin/env python3
"""
Test the new DAO model on the file with actual oscillations around zero.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

def new_dao_transfer_function(k, alpha, beta, gamma, k_star, delta, k_D, m, phi_0, r_s, epsilon):
    """New DAO transfer function model."""
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Envelope function: T_env(k) = [1 + (αk)^β]^γ
    T_env = (1 + (alpha * k_safe)**beta)**gamma
    
    # Transition function: S(k) = 0.5[1 + tanh(ln(k/k_star)/Δ)]
    S = 0.5 * (1 + np.tanh(np.log(k_safe / k_star) / delta))
    
    # Amplitude function: A(k) = T_env(k) * exp[-(k/k_D)^m]
    A = T_env * np.exp(-(k_safe / k_D)**m)
    
    # Phase function: Φ(k) = φ_0 + ∫[k_star to k] ω(u) du
    # where ω(u) = r_s[1 + ε*ln(u/k_star)]
    
    # For now, use a simpler phase function to avoid integration issues
    # Φ(k) = φ_0 + r_s * k * (1 + ε * np.log(k / k_star))
    Phi = phi_0 + r_s * k_safe * (1 + epsilon * np.log(k_safe / k_star))
    
    # Final transfer function
    T = (1 - S) * T_env + S * A * np.sin(Phi)
    
    return T

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

def test_new_model_on_oscillatory():
    """Test the new DAO model on the oscillatory file."""
    
    # Load data
    print("Loading data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Use the best oscillatory file we found
    sample_file = 'output/pk_m4.6e-04_sig1.0e-25_np2_tk.dat'
    print(f"Testing new model on: {os.path.basename(sample_file)}")
    
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
    print(f"T std: {np.std(T_filtered):.4f}")
    
    # Initial parameter guesses for the new model
    initial_guesses = [
        # Guess 1: Moderate parameters
        [0.1, 1.5, -0.5, 0.5, 0.2, 1.0, 2.0, 0.0, 1.0, 0.1],
        # Guess 2: Different scales
        [0.2, 2.0, -0.3, 0.3, 0.1, 0.8, 1.5, 0.5, 0.8, 0.05],
        # Guess 3: Conservative
        [0.15, 1.0, -0.4, 0.4, 0.15, 1.2, 2.5, 0.0, 1.2, 0.08],
        # Guess 4: Try different transition point
        [0.1, 1.2, -0.3, 0.2, 0.1, 0.5, 1.0, 0.0, 0.5, 0.0],
    ]
    
    # Parameter bounds
    bounds = (
        [0.001, 0.1, -2.0, 0.01, 0.01, 0.01, 0.1, -np.pi, 0.01, -1.0],  # Lower bounds
        [1.0, 5.0, 0.0, 2.0, 1.0, 10.0, 10.0, np.pi, 10.0, 1.0]       # Upper bounds
    )
    
    print("\nNew DAO model parameters:")
    print("α, β, γ: envelope function")
    print("k_star: transition scale")
    print("δ: transition width")
    print("k_D, m: damping parameters")
    print("φ_0: initial phase")
    print("r_s: sound horizon")
    print("ε: frequency evolution")
    
    best_fit = None
    best_r2 = -np.inf
    best_params = None
    
    for i, initial_guess in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}: {initial_guess}")
        
        try:
            popt, pcov = curve_fit(new_dao_transfer_function, k_filtered, T_filtered, 
                                  p0=initial_guess, bounds=bounds, maxfev=50000)
            
            # Calculate R²
            T_pred = new_dao_transfer_function(k_filtered, *popt)
            ss_res = np.sum((T_filtered - T_pred)**2)
            ss_tot = np.sum((T_filtered - np.mean(T_filtered))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"  R² = {r2:.4f}")
            print(f"  Parameters: α={popt[0]:.3f}, β={popt[1]:.3f}, γ={popt[2]:.3f}")
            print(f"             k_star={popt[3]:.3f}, δ={popt[4]:.3f}, k_D={popt[5]:.3f}")
            print(f"             m={popt[6]:.3f}, φ_0={popt[7]:.3f}, r_s={popt[8]:.3f}, ε={popt[9]:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = popt
                best_fit = (popt, pcov, r2)
                
        except Exception as e:
            print(f"  Fit failed: {e}")
    
    if best_fit is None:
        print("All fits failed!")
        return
    
    print(f"\nBest new DAO model fit R² = {best_r2:.4f}")
    
    # Plot the results
    plt.figure(figsize=(16, 12))
    
    # Main plot
    plt.subplot(2, 3, 1)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    T_smooth = new_dao_transfer_function(k_smooth, *best_params)
    plt.semilogx(k_smooth, T_smooth, 'r-', linewidth=2.5, 
                label=f'New DAO Model (R² = {best_r2:.4f})')
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=1, label='Zero line')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title(f'New DAO Model: {os.path.basename(sample_file)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Components breakdown
    alpha, beta, gamma, k_star, delta, k_D, m, phi_0, r_s, epsilon = best_params
    
    # Calculate components
    T_env = (1 + (alpha * k_smooth)**beta)**gamma
    S = 0.5 * (1 + np.tanh(np.log(k_smooth / k_star) / delta))
    A = T_env * np.exp(-(k_smooth / k_D)**m)
    Phi = phi_0 + r_s * k_smooth * (1 + epsilon * np.log(k_smooth / k_star))
    
    # Plot envelope
    plt.subplot(2, 3, 2)
    plt.semilogx(k_smooth, T_env, 'b-', linewidth=2, label='T_env(k)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No effect')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_env(k)')
    plt.title(f'Envelope: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot transition function
    plt.subplot(2, 3, 3)
    plt.semilogx(k_smooth, S, 'g-', linewidth=2, label='S(k)')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('S(k)')
    plt.title(f'Transition: k_star={k_star:.3f}, δ={delta:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot amplitude
    plt.subplot(2, 3, 4)
    plt.semilogx(k_smooth, A, 'purple', linewidth=2, label='A(k)')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('A(k)')
    plt.title(f'Amplitude: k_D={k_D:.3f}, m={m:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot phase
    plt.subplot(2, 3, 5)
    plt.semilogx(k_smooth, Phi, 'orange', linewidth=2, label='Φ(k)')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Φ(k)')
    plt.title(f'Phase: φ_0={phi_0:.3f}, r_s={r_s:.3f}, ε={epsilon:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 3, 6)
    T_pred = new_dao_transfer_function(k_filtered, *best_params)
    residuals = T_filtered - T_pred
    plt.semilogx(k_filtered, residuals, 'o', alpha=0.6, markersize=3, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Residuals')
    plt.title(f'Residuals (RMS = {np.sqrt(np.mean(residuals**2)):.4f})')
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'New DAO Model Parameters:\nα={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}\nk_star={k_star:.3f}, δ={delta:.3f}, k_D={k_D:.3f}\nm={m:.3f}, φ_0={phi_0:.3f}, r_s={r_s:.3f}, ε={epsilon:.3f}'
    
    plt.figtext(0.02, 0.02, param_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('new_dao_model_oscillatory_fit.png', dpi=300, bbox_inches='tight')
    print("New DAO model fit saved as new_dao_model_oscillatory_fit.png")
    plt.show()
    
    return best_params, best_r2

if __name__ == "__main__":
    print("Testing the new DAO model on oscillatory file...")
    print("=" * 60)
    
    result = test_new_model_on_oscillatory()
    
    if result is not None:
        params, r2 = result
        print(f"\nNew DAO model testing completed!")
        print(f"Best R² = {r2:.4f}")
        print("This should better capture oscillations around zero!")

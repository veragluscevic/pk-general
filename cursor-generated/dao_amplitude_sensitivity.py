#!/usr/bin/env python3
"""
Test DAO amplitude sensitivity - vary A wildly while keeping other parameters fixed.
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

def test_dao_amplitude_sensitivity():
    """Test how varying A affects the fit while keeping other parameters fixed."""
    
    # Load data
    print("Loading data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    sample_file = tk_files[10]
    print(f"Testing file: {os.path.basename(sample_file)}")
    
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
    
    # Best-fit parameters from the soft DAO fit
    best_params = [0.070, 3.000, -2.000, 0.155, 8.443, 0.345, -0.072, 0.010]
    alpha, beta, gamma, rs, kD, A_best, phi, m = best_params
    
    print(f"\nBest-fit parameters:")
    print(f"α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")
    print(f"r_s={rs:.3f}, k_D={kD:.3f}, A={A_best:.3f}, φ={phi:.3f}, m={m:.3f}")
    
    # Test wildly different A values
    A_values = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    # Create smooth k array for plotting
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    
    # Calculate envelope function (independent of A)
    T_env = (1 + (alpha * k_smooth)**beta)**gamma
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Data and fits with different A values
    plt.subplot(2, 2, 1)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(A_values)))
    
    for i, A_test in enumerate(A_values):
        T_fit = dao_transfer_function(k_smooth, alpha, beta, gamma, rs, kD, A_test, phi, m)
        plt.semilogx(k_smooth, T_fit, '-', linewidth=2, alpha=0.8,
                    label=f'A = {A_test:.0f}', color=colors[i])
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title('DAO Amplitude Sensitivity Test')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Envelope function
    plt.subplot(2, 2, 2)
    plt.semilogx(k_smooth, T_env, 'b-', linewidth=2, label='T_env(k)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No effect')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_env(k)')
    plt.title(f'Envelope: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: DAO functions for different A values
    plt.subplot(2, 2, 3)
    for i, A_test in enumerate(A_values):
        T_DAO = 1 + A_test * np.sin(k_smooth * rs + phi) * np.exp(-(k_smooth / kD)**m)
        plt.semilogx(k_smooth, T_DAO, '-', linewidth=2, alpha=0.8,
                    label=f'A = {A_test:.0f}', color=colors[i])
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No oscillations')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_DAO(k)')
    plt.title(f'DAO: r_s={rs:.3f}, k_D={kD:.3f}, φ={phi:.3f}, m={m:.3f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: R² calculation for different A values
    plt.subplot(2, 2, 4)
    r2_values = []
    
    for A_test in A_values:
        T_pred = dao_transfer_function(k_filtered, alpha, beta, gamma, rs, kD, A_test, phi, m)
        ss_res = np.sum((T_filtered - T_pred)**2)
        ss_tot = np.sum((T_filtered - np.mean(T_filtered))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        r2_values.append(r2)
    
    plt.semilogx(A_values, r2_values, 'ro-', linewidth=2, markersize=8)
    plt.axvline(x=A_best, color='green', linestyle='--', alpha=0.7, 
                label=f'Best A = {A_best:.3f}')
    plt.xlabel('DAO Amplitude A')
    plt.ylabel('R²')
    plt.title('R² vs DAO Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'Best-fit parameters (A varied):\nα={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}\nr_s={rs:.3f}, k_D={kD:.3f}, φ={phi:.3f}, m={m:.3f}'
    
    plt.figtext(0.02, 0.02, param_text, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('dao_amplitude_sensitivity.png', dpi=300, bbox_inches='tight')
    print("DAO amplitude sensitivity plot saved as dao_amplitude_sensitivity.png")
    plt.show()
    
    # Print R² values
    print(f"\nR² values for different A:")
    for i, (A_test, r2) in enumerate(zip(A_values, r2_values)):
        print(f"A = {A_test:6.0f}: R² = {r2:.4f}")
    
    print(f"\nBest A = {A_best:.3f} gave R² = 0.9275")
    
    return A_values, r2_values

if __name__ == "__main__":
    print("Testing DAO amplitude sensitivity...")
    print("=" * 50)
    
    result = test_dao_amplitude_sensitivity()
    
    if result is not None:
        A_values, r2_values = result
        print(f"\nAmplitude sensitivity test completed!")
        print("This shows how the envelope function affects the required DAO amplitude.")

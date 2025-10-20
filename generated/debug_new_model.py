#!/usr/bin/env python3
"""
Debug the new DAO model implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
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
    
    # For debugging, let's use a simpler phase function first
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

def debug_new_model():
    """Debug the new DAO model."""
    
    # Load data
    print("Loading data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Find a file with oscillations (np2)
    tk_files = glob.glob('output/*tk.dat')
    np2_files = [f for f in tk_files if 'np2' in f and 'lcdm' not in f.lower()]
    
    sample_file = np2_files[0]
    print(f"Debugging on: {os.path.basename(sample_file)}")
    
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
    
    # Test with simple parameters
    print("\nTesting with simple parameters...")
    
    # Simple test parameters
    alpha, beta, gamma = 0.1, 1.0, -0.5
    k_star, delta = 0.5, 0.2
    k_D, m = 1.0, 2.0
    phi_0, r_s, epsilon = 0.0, 1.0, 0.0
    
    params = [alpha, beta, gamma, k_star, delta, k_D, m, phi_0, r_s, epsilon]
    print(f"Test parameters: {params}")
    
    try:
        T_test = new_dao_transfer_function(k_filtered, *params)
        print(f"Model output range: {T_test.min():.4f} to {T_test.max():.4f}")
        print(f"Model output std: {np.std(T_test):.4f}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(T_test)) or np.any(np.isinf(T_test)):
            print("WARNING: Model output contains NaN or infinite values!")
            nan_mask = np.isnan(T_test) | np.isinf(T_test)
            print(f"Number of problematic points: {np.sum(nan_mask)}")
        
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        return
    
    # Plot the test
    plt.figure(figsize=(15, 10))
    
    # Main plot
    plt.subplot(2, 3, 1)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    plt.semilogx(k_filtered, T_test, 'r-', linewidth=2, 
                label='Test Model', alpha=0.8)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=1, label='Zero line')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title('Debug: Test Model vs Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot individual components
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    
    T_env = (1 + (alpha * k_smooth)**beta)**gamma
    S = 0.5 * (1 + np.tanh(np.log(k_smooth / k_star) / delta))
    A = T_env * np.exp(-(k_smooth / k_D)**m)
    Phi = phi_0 + r_s * k_smooth * (1 + epsilon * np.log(k_smooth / k_star))
    T_smooth = (1 - S) * T_env + S * A * np.sin(Phi)
    
    # Envelope
    plt.subplot(2, 3, 2)
    plt.semilogx(k_smooth, T_env, 'b-', linewidth=2, label='T_env(k)')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No effect')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_env(k)')
    plt.title('Envelope Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Transition function
    plt.subplot(2, 3, 3)
    plt.semilogx(k_smooth, S, 'g-', linewidth=2, label='S(k)')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('S(k)')
    plt.title('Transition Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Amplitude
    plt.subplot(2, 3, 4)
    plt.semilogx(k_smooth, A, 'purple', linewidth=2, label='A(k)')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('A(k)')
    plt.title('Amplitude Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Phase
    plt.subplot(2, 3, 5)
    plt.semilogx(k_smooth, Phi, 'orange', linewidth=2, label='Φ(k)')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Φ(k)')
    plt.title('Phase Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Full model
    plt.subplot(2, 3, 6)
    plt.semilogx(k_smooth, T_smooth, 'r-', linewidth=2, label='Full Model')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=1, label='Zero line')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T(k)')
    plt.title('Complete Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_new_model.png', dpi=300, bbox_inches='tight')
    print("Debug plot saved as debug_new_model.png")
    plt.show()
    
    # Calculate R² for this test
    ss_res = np.sum((T_filtered - T_test)**2)
    ss_tot = np.sum((T_filtered - np.mean(T_filtered))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    print(f"Test R² = {r2:.4f}")

if __name__ == "__main__":
    print("Debugging the new DAO model...")
    print("=" * 40)
    
    debug_new_model()
    
    print("\nDebug completed!")

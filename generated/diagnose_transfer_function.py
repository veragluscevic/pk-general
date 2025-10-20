#!/usr/bin/env python3
"""
Diagnose what's wrong with the transfer function data.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_transfer_function_data(filename):
    """Load transfer function data from CLASS output."""
    try:
        data = np.loadtxt(filename, skiprows=9)
        k = data[:, 0]
        
        print(f"Data shape: {data.shape}")
        print(f"Columns: {data.shape[1]}")
        
        # Handle different column structures
        if data.shape[1] == 9:
            d_tot = data[:, 6]  # IDM files
            print("Using column 6 (IDM file)")
        elif data.shape[1] == 8:
            d_tot = data[:, 5]  # LCDM files
            print("Using column 5 (LCDM file)")
        else:
            raise ValueError(f"Unexpected number of columns: {data.shape[1]}")
        
        return k, d_tot
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def diagnose_data():
    """Diagnose the transfer function data."""
    
    # Check LCDM file
    print("=== LCDM FILE ===")
    lcdm_file = 'output/pk-lcdm_tk.dat'
    k_lcdm, d_tot_lcdm = load_transfer_function_data(lcdm_file)
    if k_lcdm is not None:
        print(f"LCDM k range: {k_lcdm.min():.4f} to {k_lcdm.max():.4f}")
        print(f"LCDM d_tot range: {d_tot_lcdm.min():.4f} to {d_tot_lcdm.max():.4f}")
        print(f"LCDM first 5 values: {d_tot_lcdm[:5]}")
        print(f"LCDM last 5 values: {d_tot_lcdm[-5:]}")
    
    print("\n=== IDM FILE ===")
    # Find IDM file
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    sample_file = tk_files[10]
    
    print(f"IDM file: {os.path.basename(sample_file)}")
    k_idm, d_tot_idm = load_transfer_function_data(sample_file)
    if k_idm is not None:
        print(f"IDM k range: {k_idm.min():.4f} to {k_idm.max():.4f}")
        print(f"IDM d_tot range: {d_tot_idm.min():.4f} to {d_tot_idm.max():.4f}")
        print(f"IDM first 5 values: {d_tot_idm[:5]}")
        print(f"IDM last 5 values: {d_tot_idm[-5:]}")
    
    print("\n=== NORMALIZATION ===")
    # Check normalization
    if k_lcdm is not None and k_idm is not None:
        d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
        T_normalized = d_tot_idm / d_tot_lcdm_interp
        
        print(f"Normalized T range: {T_normalized.min():.4f} to {T_normalized.max():.4f}")
        print(f"Normalized T first 5 values: {T_normalized[:5]}")
        print(f"Normalized T last 5 values: {T_normalized[-5:]}")
        
        # Check for problematic values
        negative_mask = T_normalized < 0
        if np.any(negative_mask):
            print(f"Found {np.sum(negative_mask)} negative values!")
            print(f"Negative values: {T_normalized[negative_mask][:10]}")
            print(f"Corresponding k values: {k_idm[negative_mask][:10]}")
        
        # Check for very large values
        large_mask = np.abs(T_normalized) > 10
        if np.any(large_mask):
            print(f"Found {np.sum(large_mask)} very large values!")
            print(f"Large values: {T_normalized[large_mask][:10]}")

def plot_raw_data():
    """Plot the raw data before normalization."""
    
    # Load LCDM data
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Load IDM data
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    sample_file = tk_files[10]
    k_idm, d_tot_idm = load_transfer_function_data(sample_file)
    
    # Plot raw data
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogx(k_lcdm, d_tot_lcdm, 'o-', alpha=0.7, markersize=3, linewidth=1, 
                label='LCDM d_tot', color='black')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('d_tot')
    plt.title('Raw LCDM Transfer Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.semilogx(k_idm, d_tot_idm, 'o-', alpha=0.7, markersize=3, linewidth=1, 
                label='IDM d_tot', color='red')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('d_tot')
    plt.title('Raw IDM Transfer Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('raw_transfer_functions.png', dpi=300, bbox_inches='tight')
    print("Raw transfer functions saved as raw_transfer_functions.png")
    plt.show()

if __name__ == "__main__":
    print("Diagnosing transfer function data...")
    print("=" * 50)
    
    diagnose_data()
    print("\n" + "=" * 50)
    plot_raw_data()

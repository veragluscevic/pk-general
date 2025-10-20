#!/usr/bin/env python3
"""
Show the specific transfer function that we're trying to fit.
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

def show_target_transfer_function():
    """Show the specific transfer function we're trying to fit."""
    
    # Load LCDM data
    print("Loading LCDM data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    if k_lcdm is None:
        print("Failed to load LCDM data")
        return
    
    # Find the same file we used for fitting
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    # Pick the same file as before (11th file)
    sample_file = tk_files[10]
    print(f"Showing file: {os.path.basename(sample_file)}")
    
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
    
    print(f"Data points: {len(k_filtered)}")
    print(f"k range: {k_filtered.min():.4f} to {k_filtered.max():.4f} h/Mpc")
    print(f"T range: {T_filtered.min():.4f} to {T_filtered.max():.4f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the data
    plt.semilogx(k_filtered, T_filtered, 'o-', alpha=0.7, markersize=4, linewidth=1, 
                label='T_IDM(k) / T_ΛCDM(k)', color='blue')
    
    # Add reference line
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM reference')
    
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title(f'Transfer Function to Fit\n{os.path.basename(sample_file)}', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Data points: {len(k_filtered)}\n'
    stats_text += f'Mean: {np.mean(T_filtered):.4f}\n'
    stats_text += f'Std: {np.std(T_filtered):.4f}\n'
    stats_text += f'Min: {np.min(T_filtered):.4f}\n'
    stats_text += f'Max: {np.max(T_filtered):.4f}\n'
    stats_text += f'Range: {np.max(T_filtered) - np.min(T_filtered):.4f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('target_transfer_function.png', dpi=300, bbox_inches='tight')
    print("Target transfer function saved as target_transfer_function.png")
    plt.show()
    
    return k_filtered, T_filtered

if __name__ == "__main__":
    print("Showing the transfer function we're trying to fit...")
    print("=" * 50)
    
    k, T = show_target_transfer_function()
    
    print(f"\nThis is the data we're trying to fit with our enhanced function:")
    print(f"T_IDM(k) / T_ΛCDM(k) = 1 + A × exp(-(k/k_c)^α) + A_osc(k) × cos(2πk/k_osc(k) + φ(k))")

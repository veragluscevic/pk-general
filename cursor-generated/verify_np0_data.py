#!/usr/bin/env python3
"""
Verify what data we're actually plotting for the np0 file.
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

def verify_np0_data():
    """Verify what data we're actually plotting for np0 files."""
    
    print("Loading LCDM data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Find np0 files
    tk_files = glob.glob('output/*tk.dat')
    np0_files = [f for f in tk_files if 'np0' in f and 'lcdm' not in f.lower()]
    
    print(f"Found {len(np0_files)} np0 files")
    
    # Show first few np0 files
    for i, np0_file in enumerate(np0_files[:3]):
        print(f"\nFile {i+1}: {os.path.basename(np0_file)}")
        
        k_idm, d_tot_idm = load_transfer_function_data(np0_file)
        
        print(f"  Raw IDM d_tot range: {d_tot_idm.min():.4f} to {d_tot_idm.max():.4f}")
        print(f"  Raw LCDM d_tot range: {d_tot_lcdm.min():.4f} to {d_tot_lcdm.max():.4f}")
        
        # Normalize by LCDM
        d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
        T_normalized = d_tot_idm / d_tot_lcdm_interp
        
        print(f"  Normalized T range: {T_normalized.min():.4f} to {T_normalized.max():.4f}")
        
        # Filter for k > 0.01
        mask = k_idm > 0.01
        k_filtered = k_idm[mask]
        T_filtered = T_normalized[mask]
        
        print(f"  Filtered T range: {T_filtered.min():.4f} to {T_filtered.max():.4f}")
        print(f"  Data points: {len(T_filtered)}")
        
        # Check for oscillations by looking at standard deviation
        T_std = np.std(T_filtered)
        print(f"  Standard deviation: {T_std:.4f}")
        
        # Check if there are significant oscillations by looking at local extrema
        # Find local maxima and minima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(T_filtered)
        valleys, _ = find_peaks(-T_filtered)
        
        if len(peaks) > 0 and len(valleys) > 0:
            peak_values = T_filtered[peaks]
            valley_values = T_filtered[valleys]
            oscillation_amplitude = (np.mean(peak_values) - np.mean(valley_values)) / 2
            print(f"  Oscillation amplitude: {oscillation_amplitude:.4f}")
        else:
            print(f"  Oscillation amplitude: minimal (no clear peaks/valleys)")
    
    # Plot the first np0 file to show what we're actually fitting
    sample_file = np0_files[0]
    print(f"\nPlotting: {os.path.basename(sample_file)}")
    
    k_idm, d_tot_idm = load_transfer_function_data(sample_file)
    
    # Normalize by LCDM
    d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
    T_normalized = d_tot_idm / d_tot_lcdm_interp
    
    # Filter for k > 0.01
    mask = k_idm > 0.01
    k_filtered = k_idm[mask]
    T_filtered = T_normalized[mask]
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Raw IDM and LCDM data
    plt.subplot(2, 2, 1)
    plt.semilogx(k_lcdm, d_tot_lcdm, 'b-', linewidth=2, label='LCDM d_tot', alpha=0.7)
    plt.semilogx(k_idm, d_tot_idm, 'r-', linewidth=2, label='IDM d_tot', alpha=0.7)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('d_tot (raw)')
    plt.title('Raw Transfer Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Normalized data (what we actually fit)
    plt.subplot(2, 2, 2)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Normalized Data (IDM/LCDM)', color='black')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM reference')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title(f'Normalized Data: {os.path.basename(sample_file)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Zoom in on normalized data
    plt.subplot(2, 2, 3)
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.8, markersize=5, 
                label='Normalized Data', color='black')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM reference')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title('Zoomed View - Should Show Minimal Oscillations')
    plt.ylim(0.5, 1.2)  # Zoom in on the relevant range
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Compare with an np2 file for reference
    np2_files = [f for f in tk_files if 'np2' in f and 'lcdm' not in f.lower()]
    if np2_files:
        np2_file = np2_files[0]
        print(f"Comparing with np2 file: {os.path.basename(np2_file)}")
        
        k_np2, d_tot_np2 = load_transfer_function_data(np2_file)
        d_tot_lcdm_interp_np2 = np.interp(k_np2, k_lcdm, d_tot_lcdm)
        T_normalized_np2 = d_tot_np2 / d_tot_lcdm_interp_np2
        
        mask_np2 = k_np2 > 0.01
        k_filtered_np2 = k_np2[mask_np2]
        T_filtered_np2 = T_normalized_np2[mask_np2]
        
        plt.subplot(2, 2, 4)
        plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                    label=f'np0: {os.path.basename(sample_file)}', color='blue')
        plt.semilogx(k_filtered_np2, T_filtered_np2, 'o', alpha=0.6, markersize=4, 
                    label=f'np2: {os.path.basename(np2_file)}', color='red')
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM reference')
        plt.xlabel('k [h/Mpc]')
        plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
        plt.title('np0 vs np2 Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('np0_data_verification.png', dpi=300, bbox_inches='tight')
    print("Data verification plot saved as np0_data_verification.png")
    plt.show()
    
    return sample_file, k_filtered, T_filtered

if __name__ == "__main__":
    print("Verifying np0 data - what are we actually plotting?")
    print("=" * 60)
    
    result = verify_np0_data()
    
    if result is not None:
        sample_file, k_filtered, T_filtered = result
        print(f"\nData verification completed!")
        print("This should clarify what data points we're actually fitting.")

#!/usr/bin/env python3
"""
Find a file that actually has oscillations around zero.
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

def find_oscillatory_file():
    """Find a file that actually has oscillations around zero."""
    
    # Load LCDM data
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Find all transfer function files
    tk_files = glob.glob('output/*tk.dat')
    idm_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    print(f"Found {len(idm_files)} IDM files")
    print("Searching for files with oscillations around zero...")
    
    # Test files to find one with significant oscillations
    best_files = []
    
    for i, file in enumerate(idm_files[:20]):  # Test first 20 files
        k_idm, d_tot_idm = load_transfer_function_data(file)
        
        # Normalize by LCDM
        d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
        T_normalized = d_tot_idm / d_tot_lcdm_interp
        
        # Filter for k > 0.01
        mask = k_idm > 0.01
        T_filtered = T_normalized[mask]
        
        # Look for files that cross zero or have large negative values
        min_T = T_filtered.min()
        max_T = T_filtered.max()
        std_T = np.std(T_filtered)
        
        # Score based on how much it deviates from 1.0
        deviation_score = max(abs(min_T - 1.0), abs(max_T - 1.0))
        
        if min_T < 0.5 or max_T > 1.5 or std_T > 0.1:  # Significant deviations
            best_files.append((file, min_T, max_T, std_T, deviation_score))
            print(f"File {i+1}: {os.path.basename(file)}")
            print(f"  T range: {min_T:.4f} to {max_T:.4f}")
            print(f"  T std: {std_T:.4f}")
            print(f"  Deviation score: {deviation_score:.4f}")
            print()
    
    if not best_files:
        print("No files with significant oscillations found in first 20 files.")
        print("Let's look at a few more...")
        
        # Look at more files
        for i, file in enumerate(idm_files[20:40]):
            k_idm, d_tot_idm = load_transfer_function_data(file)
            d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
            T_normalized = d_tot_idm / d_tot_lcdm_interp
            mask = k_idm > 0.01
            T_filtered = T_normalized[mask]
            
            min_T = T_filtered.min()
            max_T = T_filtered.max()
            std_T = np.std(T_filtered)
            deviation_score = max(abs(min_T - 1.0), abs(max_T - 1.0))
            
            if min_T < 0.8 or max_T > 1.2 or std_T > 0.05:
                best_files.append((file, min_T, max_T, std_T, deviation_score))
                print(f"File {i+21}: {os.path.basename(file)}")
                print(f"  T range: {min_T:.4f} to {max_T:.4f}")
                print(f"  T std: {std_T:.4f}")
                print()
    
    if best_files:
        # Sort by deviation score
        best_files.sort(key=lambda x: x[4], reverse=True)
        
        print(f"Best files with oscillations:")
        for i, (file, min_T, max_T, std_T, score) in enumerate(best_files[:5]):
            print(f"{i+1}. {os.path.basename(file)} (score: {score:.4f})")
        
        # Plot the best file
        best_file = best_files[0][0]
        print(f"\nPlotting best file: {os.path.basename(best_file)}")
        
        k_idm, d_tot_idm = load_transfer_function_data(best_file)
        d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
        T_normalized = d_tot_idm / d_tot_lcdm_interp
        mask = k_idm > 0.01
        k_filtered = k_idm[mask]
        T_filtered = T_normalized[mask]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                    label='Data', color='black')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM reference')
        plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Zero line')
        plt.xlabel('k [h/Mpc]')
        plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
        plt.title(f'Best Oscillatory File: {os.path.basename(best_file)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.8, markersize=5, 
                    label='Data', color='black')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM reference')
        plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Zero line')
        plt.xlabel('k [h/Mpc]')
        plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
        plt.title('Zoomed View')
        plt.ylim(-0.5, 1.5)  # Show the full range
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('best_oscillatory_file.png', dpi=300, bbox_inches='tight')
        print("Best oscillatory file plot saved as best_oscillatory_file.png")
        plt.show()
        
        return best_file, k_filtered, T_filtered
    else:
        print("No files with significant oscillations found!")
        return None, None, None

if __name__ == "__main__":
    print("Finding files with oscillations around zero...")
    print("=" * 50)
    
    result = find_oscillatory_file()
    
    if result[0] is not None:
        best_file, k_filtered, T_filtered = result
        print(f"\nFound oscillatory file: {os.path.basename(best_file)}")
        print(f"T range: {T_filtered.min():.4f} to {T_filtered.max():.4f}")
    else:
        print("\nNo suitable oscillatory files found.")

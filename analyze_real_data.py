#!/usr/bin/env python3
"""
Analyze real transfer function data to understand its characteristics
before fitting the enhanced function.
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

def analyze_data_characteristics():
    """Analyze the characteristics of real transfer function data."""
    
    # Load LCDM data
    print("Loading LCDM data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    if k_lcdm is None:
        print("Failed to load LCDM data")
        return
    
    # Find sample IDM files
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    print(f"Found {len(tk_files)} IDM files")
    
    # Analyze a few sample files
    sample_files = tk_files[:5]  # First 5 files
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Real Transfer Function Data Analysis\nNormalized by LCDM', fontsize=14, y=0.98)
    
    for i, filename in enumerate(sample_files):
        if i >= 5:  # Only plot first 5
            break
            
        ax = axes[i//3, i%3]
        
        # Load IDM data
        k_idm, d_tot_idm = load_transfer_function_data(filename)
        if k_idm is None:
            continue
        
        # Normalize by LCDM
        d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
        T_normalized = d_tot_idm / d_tot_lcdm_interp
        
        # Filter for k > 0.01
        mask = k_idm > 0.01
        k_filtered = k_idm[mask]
        T_filtered = T_normalized[mask]
        
        # Plot data
        ax.semilogx(k_filtered, T_filtered, 'o-', alpha=0.7, markersize=3, linewidth=1)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
        
        # Add statistics
        mean_val = np.mean(T_filtered)
        std_val = np.std(T_filtered)
        min_val = np.min(T_filtered)
        max_val = np.max(T_filtered)
        
        ax.set_xlabel('k [h/Mpc]')
        ax.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
        ax.set_title(f'{os.path.basename(filename)[:25]}...\nMean: {mean_val:.3f}, Std: {std_val:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Min: {min_val:.3f}\nMax: {max_val:.3f}\nRange: {max_val-min_val:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=8)
    
    # Remove empty subplot
    if len(sample_files) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('real_data_analysis.png', dpi=300, bbox_inches='tight')
    print("Real data analysis saved as real_data_analysis.png")
    plt.show()

def plot_detailed_single_file():
    """Plot detailed analysis of a single file."""
    
    # Load LCDM data
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Pick a specific file
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    sample_file = tk_files[10]  # Same as before
    
    print(f"Detailed analysis of: {os.path.basename(sample_file)}")
    
    # Load IDM data
    k_idm, d_tot_idm = load_transfer_function_data(sample_file)
    
    # Normalize by LCDM
    d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
    T_normalized = d_tot_idm / d_tot_lcdm_interp
    
    # Filter for k > 0.01
    mask = k_idm > 0.01
    k_filtered = k_idm[mask]
    T_filtered = T_normalized[mask]
    
    # Create detailed plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Detailed Analysis: {os.path.basename(sample_file)}', fontsize=14)
    
    # Plot 1: Full data
    ax1 = axes[0, 0]
    ax1.semilogx(k_filtered, T_filtered, 'o-', alpha=0.7, markersize=4, linewidth=1)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax1.set_title('Full Transfer Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed in on oscillations
    ax2 = axes[0, 1]
    # Focus on a specific range where oscillations are visible
    zoom_mask = (k_filtered > 0.1) & (k_filtered < 2.0)
    if np.any(zoom_mask):
        ax2.semilogx(k_filtered[zoom_mask], T_filtered[zoom_mask], 'o-', alpha=0.8, markersize=4, linewidth=1)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
        ax2.set_xlabel('k [h/Mpc]')
        ax2.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
        ax2.set_title('Zoomed: Oscillation Region')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data in zoom range', transform=ax2.transAxes, ha='center', va='center')
    
    # Plot 3: Deviation from LCDM
    ax3 = axes[1, 0]
    deviation = T_filtered - 1
    ax3.semilogx(k_filtered, deviation, 'o-', alpha=0.7, markersize=3, linewidth=1, color='red')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No deviation')
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('Deviation from ΛCDM')
    ax3.set_title('Deviation from ΛCDM')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    stats = {
        'Mean': np.mean(T_filtered),
        'Std': np.std(T_filtered),
        'Min': np.min(T_filtered),
        'Max': np.max(T_filtered),
        'Range': np.max(T_filtered) - np.min(T_filtered),
        'Data points': len(T_filtered),
        'k range': f'{k_filtered.min():.3f} - {k_filtered.max():.3f}'
    }
    
    stats_text = '\n'.join([f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}' 
                           for key, value in stats.items()])
    
    ax4.text(0.1, 0.9, 'Statistics:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    ax4.text(0.1, 0.7, stats_text, fontsize=10, transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('detailed_single_file_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed single file analysis saved as detailed_single_file_analysis.png")
    plt.show()
    
    return k_filtered, T_filtered

def analyze_oscillatory_features(k, T):
    """Analyze oscillatory features in the data."""
    
    print("\nAnalyzing oscillatory features...")
    
    # Look for patterns in the deviation from LCDM
    deviation = T - 1
    
    # Calculate local extrema
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(deviation, height=0.01)
    valleys, _ = find_peaks(-deviation, height=0.01)
    
    print(f"Found {len(peaks)} peaks and {len(valleys)} valleys")
    
    if len(peaks) > 2:
        # Calculate approximate oscillation wavelength
        peak_k_values = k[peaks]
        if len(peak_k_values) > 1:
            avg_wavelength = np.mean(np.diff(peak_k_values))
            print(f"Average oscillation wavelength: {avg_wavelength:.3f} h/Mpc")
    
    # Plot with extrema marked
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(k, T, 'o-', alpha=0.7, markersize=3, linewidth=1, label='Data')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    if len(peaks) > 0:
        plt.semilogx(k[peaks], T[peaks], 'ro', markersize=6, label='Peaks')
    if len(valleys) > 0:
        plt.semilogx(k[valleys], T[valleys], 'bo', markersize=6, label='Valleys')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title('Transfer Function with Extrema')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(k, deviation, 'o-', alpha=0.7, markersize=3, linewidth=1, color='red', label='Deviation')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No deviation')
    if len(peaks) > 0:
        plt.semilogx(k[peaks], deviation[peaks], 'ro', markersize=6, label='Peaks')
    if len(valleys) > 0:
        plt.semilogx(k[valleys], deviation[valleys], 'bo', markersize=6, label='Valleys')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Deviation from ΛCDM')
    plt.title('Deviation with Extrema')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oscillatory_features_analysis.png', dpi=300, bbox_inches='tight')
    print("Oscillatory features analysis saved as oscillatory_features_analysis.png")
    plt.show()

if __name__ == "__main__":
    print("Analyzing real transfer function data characteristics...")
    print("=" * 60)
    
    # Analyze multiple files
    analyze_data_characteristics()
    
    # Detailed analysis of single file
    k, T = plot_detailed_single_file()
    
    # Analyze oscillatory features
    analyze_oscillatory_features(k, T)
    
    print("\nAnalysis complete!")
    print("This will help us understand the data characteristics before fitting.")

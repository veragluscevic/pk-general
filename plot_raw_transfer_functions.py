#!/usr/bin/env python3
"""
Plot raw transfer functions for fixed n and mass, varying cross sections.

This script plots the raw d_tot transfer functions (not normalized, not squared)
for a fixed interaction type (n) and mass, but varying cross sections,
along with the LCDM transfer function for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

def read_transfer_function(filename):
    """Read transfer function data from CLASS output file."""
    try:
        data = np.loadtxt(filename, skiprows=9)  # Skip 9 header lines
        if len(data) == 0:
            return None, None
        
        k = data[:, 0]  # k values (h/Mpc)
        d_tot = data[:, 7]  # d_tot column (total matter transfer function)
        
        return k, d_tot
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def parse_filename(filename):
    """Parse filename to extract parameters."""
    basename = os.path.basename(filename)
    
    # Extract mass (m*.*e-*)
    mass_match = re.search(r'_m([0-9.]+e[-+]?[0-9]+)_', basename)
    mass = float(mass_match.group(1)) if mass_match else None
    
    # Extract cross section (sig*.*e-*)
    sig_match = re.search(r'_sig([0-9.]+e[-+]?[0-9]+)_', basename)
    cross_section = float(sig_match.group(1)) if sig_match else None
    
    # Extract interaction type (np*)
    np_match = re.search(r'_np([0-9]+)_', basename)
    interaction_type = int(np_match.group(1)) if np_match else None
    
    return mass, cross_section, interaction_type

def main():
    # Find all transfer function files
    files = glob.glob("output/*_tk.dat")
    
    # Filter for LCDM file
    lcdm_files = [f for f in files if 'lcdm' in f.lower()]
    if not lcdm_files:
        print("No LCDM file found!")
        return
    
    # Read LCDM transfer function
    k_lcdm, d_tot_lcdm = read_transfer_function(lcdm_files[0])
    if k_lcdm is None:
        print("Failed to read LCDM file!")
        return
    
    # Filter for a specific interaction type and mass
    # Let's use np2 and a specific mass
    target_np = 2  # np2
    target_mass = 1.0e-05  # 1e-05 GeV
    
    # Find files matching our criteria
    matching_files = []
    for filename in files:
        mass, cross_section, interaction_type = parse_filename(filename)
        if (mass == target_mass and 
            interaction_type == target_np and 
            cross_section is not None):
            matching_files.append((filename, cross_section))
    
    if len(matching_files) < 3:
        print(f"Found only {len(matching_files)} files matching criteria. Need at least 3.")
        print("Available files:")
        for f, cs in matching_files:
            print(f"  Cross section: {cs}")
        return
    
    # Sort by cross section and take 3 different ones
    matching_files.sort(key=lambda x: x[1])
    
    # Select 3 cross sections that are closer in value and as large as possible
    if len(matching_files) >= 3:
        # Take the 3 largest cross sections (closest to each other at the high end)
        selected_files = matching_files[-3:]
    else:
        selected_files = matching_files
    
    print(f"Selected files for np{target_np}, mass={target_mass} GeV:")
    for filename, cross_section in selected_files:
        print(f"  Cross section: {cross_section:.2e} cm²")
    
    # Create three plots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
    
    # Plot 1: Raw transfer functions
    # Plot LCDM transfer function
    ax1.semilogx(k_lcdm, d_tot_lcdm, 'k-', linewidth=2, label='LCDM', alpha=0.8)
    
    # Plot IDM transfer functions
    colors = ['blue', 'red', 'green']
    for i, (filename, cross_section) in enumerate(selected_files):
        k, d_tot = read_transfer_function(filename)
        if k is not None and d_tot is not None:
            # Filter for k > 0.1
            mask = k > 0.1
            k_filtered = k[mask]
            d_tot_filtered = d_tot[mask]
            
            ax1.semilogx(k_filtered, d_tot_filtered, 
                        color=colors[i], linewidth=1.5, 
                        label=f'IDM σ={cross_section:.2e} cm²')
    
    ax1.set_xlabel('k [h/Mpc]', fontsize=12)
    ax1.set_ylabel('d_tot (Transfer Function)', fontsize=12)
    ax1.set_title(f'Raw Transfer Functions: np{target_np}, m={target_mass:.0e} GeV', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.1, max(k_lcdm))
    
    # Add text box with parameters
    textstr = f'Fixed: np{target_np}, m={target_mass:.0e} GeV\nVarying: Cross section σ'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot 2: LCDM-normalized transfer functions
    # Plot reference line at 1
    ax2.semilogx(k_lcdm, np.ones_like(k_lcdm), 'k--', linewidth=1, label='LCDM reference (1.0)', alpha=0.6)
    
    # Plot normalized IDM transfer functions
    for i, (filename, cross_section) in enumerate(selected_files):
        k, d_tot = read_transfer_function(filename)
        if k is not None and d_tot is not None:
            # Filter for k > 0.1
            mask = k > 0.1
            k_filtered = k[mask]
            d_tot_filtered = d_tot[mask]
            
            # Interpolate LCDM to match filtered IDM k values
            d_tot_lcdm_interp = np.interp(k_filtered, k_lcdm, d_tot_lcdm)
            # Normalize by LCDM
            d_tot_normalized = d_tot_filtered / d_tot_lcdm_interp
            
            ax2.semilogx(k_filtered, d_tot_normalized, 
                        color=colors[i], linewidth=1.5, 
                        label=f'IDM σ={cross_section:.2e} cm²')
    
    ax2.set_xlabel('k [h/Mpc]', fontsize=12)
    ax2.set_ylabel('T_IDM(k) / T_LCDM(k)', fontsize=12)
    ax2.set_title(f'LCDM-Normalized Transfer Functions: np{target_np}, m={target_mass:.0e} GeV', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.1, max(k_lcdm))
    
    # Add text box with parameters
    textstr2 = f'Fixed: np{target_np}, m={target_mass:.0e} GeV\nVarying: Cross section σ'
    ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot 3: Square root of absolute value, normalized to LCDM
    # Plot reference line at 1
    ax3.semilogx(k_lcdm, np.ones_like(k_lcdm), 'k--', linewidth=1, label='LCDM reference (1.0)', alpha=0.6)
    
    # Plot sqrt(|T|) normalized IDM transfer functions
    for i, (filename, cross_section) in enumerate(selected_files):
        k, d_tot = read_transfer_function(filename)
        if k is not None and d_tot is not None:
            # Filter for k > 0.1
            mask = k > 0.1
            k_filtered = k[mask]
            d_tot_filtered = d_tot[mask]
            
            # Interpolate LCDM to match filtered IDM k values
            d_tot_lcdm_interp = np.interp(k_filtered, k_lcdm, d_tot_lcdm)
            # Take square root of absolute value and normalize
            sqrt_abs_d_tot = np.sqrt(np.abs(d_tot_filtered))
            sqrt_abs_d_tot_lcdm = np.sqrt(np.abs(d_tot_lcdm_interp))
            d_tot_sqrt_normalized = sqrt_abs_d_tot / sqrt_abs_d_tot_lcdm
            
            ax3.semilogx(k_filtered, d_tot_sqrt_normalized, 
                        color=colors[i], linewidth=1.5, 
                        label=f'IDM σ={cross_section:.2e} cm²')
    
    ax3.set_xlabel('k [h/Mpc]', fontsize=12)
    ax3.set_ylabel('√|T_IDM(k)| / √|T_LCDM(k)|', fontsize=12)
    ax3.set_title(f'√|T| Normalized: np{target_np}, m={target_mass:.0e} GeV', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.1, max(k_lcdm))
    
    # Add text box with parameters
    textstr3 = f'Fixed: np{target_np}, m={target_mass:.0e} GeV\nVarying: Cross section σ'
    ax3.text(0.02, 0.98, textstr3, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('transfer_functions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as 'transfer_functions_comparison.png'")
    print(f"Parameters used:")
    print(f"  Fixed interaction type: np{target_np}")
    print(f"  Fixed mass: {target_mass:.0e} GeV")
    print(f"  Varying cross sections: {[f'{cs:.2e}' for _, cs in selected_files]}")
    print(f"  Created three plots: raw transfer functions (left), LCDM-normalized (center), and √|T| normalized (right)")

if __name__ == "__main__":
    main()

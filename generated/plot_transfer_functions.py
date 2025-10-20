#!/usr/bin/env python3
"""
Plot all transfer functions from CLASS output files.

This script reads all transfer function files (*tk.dat) from the output directory
and creates a comprehensive semilogx plot showing all transfer functions.
The files contain transfer functions for different dark matter physics scenarios
with varying particle masses, cross sections, and interaction types.

File naming convention:
- pk_m{mass}_sig{cross_section}_np{interaction_type}_tk.dat
- mass: DM particle mass in GeV
- cross_section: interaction cross section in cm^2
- interaction_type: choice of interaction type (np0, np2, np4)
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from pathlib import Path

def parse_filename(filename):
    """
    Parse filename to extract parameters.
    
    Args:
        filename (str): Full path to the transfer function file
        
    Returns:
        dict: Dictionary containing parsed parameters
    """
    basename = os.path.basename(filename)
    
    # Extract parameters using regex
    pattern = r'pk_m([0-9.e+-]+)_sig([0-9.e+-]+)_np([0-9]+)_tk\.dat'
    match = re.match(pattern, basename)
    
    if match:
        mass = float(match.group(1))
        cross_section = float(match.group(2))
        interaction_type = int(match.group(3))
        
        return {
            'mass': mass,
            'cross_section': cross_section,
            'interaction_type': interaction_type,
            'filename': basename
        }
    else:
        return None

def read_transfer_function(filename):
    """
    Read transfer function data from CLASS output file.
    
    Args:
        filename (str): Path to the transfer function file
        
    Returns:
        tuple: (k, transfer_functions) where k is wavenumber and transfer_functions is dict
    """
    try:
        # Read the data, skipping header lines
        data = np.loadtxt(filename, skiprows=9)
        
        # Extract k (wavenumber) and transfer functions
        k = data[:, 0]  # k in h/Mpc
        
        # Transfer functions for different components
        # Handle both IDM files (9 columns) and LCDM files (8 columns)
        if data.shape[1] == 9:
            # IDM files have d_dmeff column
            transfer_functions = {
                'd_g': data[:, 1],      # gas/radiation
                'd_b': data[:, 2],      # baryons
                'd_cdm': data[:, 3],    # cold dark matter
                'd_dmeff': data[:, 4],  # effective dark matter
                'd_ur': data[:, 5],     # ultra-relativistic
                'd_tot': data[:, 6],    # total matter
                'phi': data[:, 7],      # gravitational potential
                'psi': data[:, 8]       # gravitational potential
            }
        elif data.shape[1] == 8:
            # LCDM files don't have d_dmeff column
            transfer_functions = {
                'd_g': data[:, 1],      # gas/radiation
                'd_b': data[:, 2],      # baryons
                'd_cdm': data[:, 3],    # cold dark matter
                'd_ur': data[:, 4],     # ultra-relativistic
                'd_tot': data[:, 5],    # total matter
                'phi': data[:, 6],      # gravitational potential
                'psi': data[:, 7]       # gravitational potential
            }
        else:
            raise ValueError(f"Unexpected number of columns: {data.shape[1]}")
        
        return k, transfer_functions
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def plot_all_transfer_functions(output_dir='output', save_plot=True):
    """
    Plot all transfer functions from the output directory, normalized by LCDM.
    
    Args:
        output_dir (str): Directory containing transfer function files
        save_plot (bool): Whether to save the plot to file
    """
    # Find all transfer function files (excluding LCDM)
    tk_files = glob.glob(os.path.join(output_dir, '*tk.dat'))
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    if not tk_files:
        print(f"No transfer function files found in {output_dir}")
        return
    
    # Load LCDM transfer function for normalization
    lcdm_file = os.path.join(output_dir, 'pk-lcdm_tk.dat')
    if not os.path.exists(lcdm_file):
        print(f"LCDM transfer function file not found: {lcdm_file}")
        return
    
    print("Loading LCDM transfer function for normalization...")
    k_lcdm, lcdm_transfer = read_transfer_function(lcdm_file)
    if k_lcdm is None:
        print("Failed to read LCDM transfer function")
        return
    
    print(f"Found {len(tk_files)} transfer function files (excluding LCDM)")
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Colors for different interaction types
    colors = {0: 'blue', 2: 'red', 4: 'green'}
    interaction_labels = {0: 'np0', 2: 'np2', 4: 'np4'}
    
    # Plot each transfer function
    plotted_count = 0
    failed_count = 0
    
    for filename in sorted(tk_files):
        # Parse filename to get parameters
        params = parse_filename(filename)
        if params is None:
            print(f"Could not parse filename: {filename}")
            failed_count += 1
            continue
        
        # Read the data
        k, transfer_functions = read_transfer_function(filename)
        if k is None:
            failed_count += 1
            continue
        
        # Create label for this transfer function
        label = f"m={params['mass']:.1e} GeV, σ={params['cross_section']:.1e} cm², {interaction_labels[params['interaction_type']]}"
        
        # Normalize by LCDM transfer function
        # Interpolate LCDM to match the k values of this file
        lcdm_tot_interp = np.interp(k, k_lcdm, lcdm_transfer['d_tot'])
        
        # Calculate normalized transfer function (IDM / LCDM)
        normalized_tf = transfer_functions['d_tot'] / lcdm_tot_interp
        
        # Filter for k > 0.01 h/Mpc
        mask = k > 0.01
        k_filtered = k[mask]
        normalized_tf_filtered = normalized_tf[mask]
        
        # Plot the normalized transfer function
        color = colors.get(params['interaction_type'], 'black')
        plt.semilogx(k_filtered, normalized_tf_filtered, 
                    color=color, alpha=0.7, linewidth=0.8, label=label)
        
        plotted_count += 1
        
        # Limit number of labels to avoid overcrowding
        if plotted_count > 20:
            plt.semilogx(k_filtered, normalized_tf_filtered, 
                        color=color, alpha=0.7, linewidth=0.8)
    
    # Customize the plot
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title('Normalized Dark Matter Transfer Functions\n(IDM / ΛCDM) - Deviations from Standard Cosmology', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=1 to show LCDM reference
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM reference')
    
    # Add legend only if we have few enough lines
    if plotted_count <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Add text box with summary
    summary_text = f'Total files: {len(tk_files)}\nPlotted: {plotted_count}\nFailed: {failed_count}'
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        output_filename = 'all_transfer_functions.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_filename}")
    
    plt.show()
    
    return plotted_count, failed_count

def plot_by_parameter(output_dir='output', parameter='mass', save_plot=True):
    """
    Plot transfer functions grouped by a specific parameter, normalized by LCDM.
    
    Args:
        output_dir (str): Directory containing transfer function files
        parameter (str): Parameter to group by ('mass', 'cross_section', 'interaction_type')
        save_plot (bool): Whether to save the plot to file
    """
    # Find all transfer function files (excluding LCDM)
    tk_files = glob.glob(os.path.join(output_dir, '*tk.dat'))
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    if not tk_files:
        print(f"No transfer function files found in {output_dir}")
        return
    
    # Load LCDM transfer function for normalization
    lcdm_file = os.path.join(output_dir, 'pk-lcdm_tk.dat')
    if not os.path.exists(lcdm_file):
        print(f"LCDM transfer function file not found: {lcdm_file}")
        return
    
    k_lcdm, lcdm_transfer = read_transfer_function(lcdm_file)
    if k_lcdm is None:
        print("Failed to read LCDM transfer function")
        return
    
    # Group files by parameter
    groups = {}
    for filename in tk_files:
        params = parse_filename(filename)
        if params is None:
            continue
        
        key = params[parameter]
        if key not in groups:
            groups[key] = []
        groups[key].append((filename, params))
    
    # Create subplots
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 6))
    if n_groups == 1:
        axes = [axes]
    
    colors = {0: 'blue', 2: 'red', 4: 'green'}
    
    for i, (key, files) in enumerate(sorted(groups.items())):
        ax = axes[i]
        
        for filename, params in files:
            k, transfer_functions = read_transfer_function(filename)
            if k is None:
                continue
            
            # Normalize by LCDM transfer function
            lcdm_tot_interp = np.interp(k, k_lcdm, lcdm_transfer['d_tot'])
            normalized_tf = transfer_functions['d_tot'] / lcdm_tot_interp
            
            # Filter for k > 0.01 h/Mpc
            mask = k > 0.01
            k_filtered = k[mask]
            normalized_tf_filtered = normalized_tf[mask]
            
            color = colors.get(params['interaction_type'], 'black')
            ax.semilogx(k_filtered, normalized_tf_filtered, 
                       color=color, alpha=0.7, linewidth=1.0)
        
        ax.set_xlabel('k [h/Mpc]')
        ax.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
        ax.set_title(f'{parameter} = {key}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle(f'Transfer Functions Grouped by {parameter.title()}', fontsize=14)
    plt.tight_layout()
    
    if save_plot:
        output_filename = f'transfer_functions_by_{parameter}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    # Main execution
    print("Plotting all transfer functions...")
    plotted, failed = plot_all_transfer_functions()
    print(f"Successfully plotted {plotted} transfer functions")
    print(f"Failed to read {failed} files")
    
    # Create plot grouped by interaction type only
    print("\nCreating plot grouped by interaction type...")
    plot_by_parameter(parameter='interaction_type')

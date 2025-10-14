#!/usr/bin/env python3
"""
Test the analytic transfer function against the IDM data.

This script loads the transfer function data, fits the analytic function,
and compares the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from analytic_transfer_function import (
    analytic_transfer_function, 
    enhanced_analytic_function,
    fit_analytic_function,
    parameter_interpretation,
    plot_fit_comparison
)

def load_transfer_function(filename):
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

def load_lcdm_transfer_function(output_dir='output'):
    """Load the LCDM transfer function for normalization."""
    lcdm_file = os.path.join(output_dir, 'pk-lcdm_tk.dat')
    k, d_tot = load_transfer_function(lcdm_file)
    return k, d_tot

def test_analytic_function_on_sample_data():
    """Test the analytic function on a sample of the data."""
    
    print("Loading LCDM transfer function...")
    k_lcdm, d_tot_lcdm = load_lcdm_transfer_function()
    if k_lcdm is None:
        print("Failed to load LCDM data")
        return
    
    # Find a few sample IDM files to test
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    
    print(f"Found {len(tk_files)} IDM files")
    
    # Test on a few representative files
    test_files = tk_files[:5]  # Test first 5 files
    
    results = []
    
    for filename in test_files:
        print(f"\nTesting file: {os.path.basename(filename)}")
        
        # Load IDM data
        k_idm, d_tot_idm = load_transfer_function(filename)
        if k_idm is None:
            continue
        
        # Normalize by LCDM
        d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
        T_normalized = d_tot_idm / d_tot_lcdm_interp
        
        # Filter for k > 0.01
        mask = k_idm > 0.01
        k_filtered = k_idm[mask]
        T_filtered = T_normalized[mask]
        
        if len(k_filtered) < 10:
            print("  Not enough data points after filtering")
            continue
        
        # Fit basic analytic function
        print("  Fitting basic analytic function...")
        popt_basic, pcov_basic = fit_analytic_function(
            k_filtered, T_filtered, function_type='basic'
        )
        
        if popt_basic is not None:
            # Calculate R²
            T_pred_basic = analytic_transfer_function(k_filtered, *popt_basic)
            r2_basic = 1 - np.sum((T_filtered - T_pred_basic)**2) / np.sum((T_filtered - np.mean(T_filtered))**2)
            
            print(f"  Basic function R² = {r2_basic:.4f}")
            print(f"  Parameters: A={popt_basic[0]:.3f}, k_c={popt_basic[1]:.3f}, α={popt_basic[2]:.3f}, β={popt_basic[3]:.3f}")
            
            results.append({
                'filename': os.path.basename(filename),
                'function_type': 'basic',
                'params': popt_basic,
                'r2': r2_basic,
                'k_data': k_filtered,
                'T_data': T_filtered
            })
        
        # Fit enhanced analytic function
        print("  Fitting enhanced analytic function...")
        popt_enhanced, pcov_enhanced = fit_analytic_function(
            k_filtered, T_filtered, function_type='enhanced'
        )
        
        if popt_enhanced is not None:
            T_pred_enhanced = enhanced_analytic_function(k_filtered, *popt_enhanced)
            r2_enhanced = 1 - np.sum((T_filtered - T_pred_enhanced)**2) / np.sum((T_filtered - np.mean(T_filtered))**2)
            
            print(f"  Enhanced function R² = {r2_enhanced:.4f}")
            print(f"  Parameters: A={popt_enhanced[0]:.3f}, k_c={popt_enhanced[1]:.3f}, α={popt_enhanced[2]:.3f}, β={popt_enhanced[3]:.3f}, δ={popt_enhanced[4]:.3f}, k_damp={popt_enhanced[5]:.3f}")
            
            results.append({
                'filename': os.path.basename(filename),
                'function_type': 'enhanced',
                'params': popt_enhanced,
                'r2': r2_enhanced,
                'k_data': k_filtered,
                'T_data': T_filtered
            })
    
    return results

def plot_best_fits(results):
    """Plot the best fits for visualization."""
    
    # Find best basic and enhanced fits
    basic_results = [r for r in results if r['function_type'] == 'basic']
    enhanced_results = [r for r in results if r['function_type'] == 'enhanced']
    
    if not basic_results and not enhanced_results:
        print("No successful fits to plot")
        return
    
    # Plot best basic fit
    if basic_results:
        best_basic = max(basic_results, key=lambda x: x['r2'])
        print(f"\nBest basic fit: {best_basic['filename']} (R² = {best_basic['r2']:.4f})")
        
        plot_fit_comparison(
            best_basic['k_data'], 
            best_basic['T_data'], 
            best_basic['params'],
            function_type='basic',
            title=f"Best Basic Fit: {best_basic['filename']}",
            save_plot=True
        )
    
    # Plot best enhanced fit
    if enhanced_results:
        best_enhanced = max(enhanced_results, key=lambda x: x['r2'])
        print(f"\nBest enhanced fit: {best_enhanced['filename']} (R² = {best_enhanced['r2']:.4f})")
        
        plot_fit_comparison(
            best_enhanced['k_data'], 
            best_enhanced['T_data'], 
            best_enhanced['params'],
            function_type='enhanced',
            title=f"Best Enhanced Fit: {best_enhanced['filename']}",
            save_plot=True
        )

def analyze_parameter_trends(results):
    """Analyze trends in the fitted parameters."""
    
    print("\n" + "="*60)
    print("PARAMETER ANALYSIS")
    print("="*60)
    
    basic_results = [r for r in results if r['function_type'] == 'basic']
    enhanced_results = [r for r in results if r['function_type'] == 'enhanced']
    
    if basic_results:
        print(f"\nBasic Function Results ({len(basic_results)} fits):")
        print("-" * 40)
        
        # Extract parameters
        A_values = [r['params'][0] for r in basic_results]
        k_c_values = [r['params'][1] for r in basic_results]
        alpha_values = [r['params'][2] for r in basic_results]
        beta_values = [r['params'][3] for r in basic_results]
        r2_values = [r['r2'] for r in basic_results]
        
        print(f"Amplitude (A): {np.mean(A_values):.3f} ± {np.std(A_values):.3f}")
        print(f"Transition scale (k_c): {np.mean(k_c_values):.3f} ± {np.std(k_c_values):.3f}")
        print(f"Low-k power (α): {np.mean(alpha_values):.3f} ± {np.std(alpha_values):.3f}")
        print(f"High-k suppression (β): {np.mean(beta_values):.3f} ± {np.std(beta_values):.3f}")
        print(f"Average R²: {np.mean(r2_values):.4f}")
    
    if enhanced_results:
        print(f"\nEnhanced Function Results ({len(enhanced_results)} fits):")
        print("-" * 40)
        
        # Extract parameters
        A_values = [r['params'][0] for r in enhanced_results]
        k_c_values = [r['params'][1] for r in enhanced_results]
        alpha_values = [r['params'][2] for r in enhanced_results]
        beta_values = [r['params'][3] for r in enhanced_results]
        delta_values = [r['params'][4] for r in enhanced_results]
        k_damp_values = [r['params'][5] for r in enhanced_results]
        r2_values = [r['r2'] for r in enhanced_results]
        
        print(f"Amplitude (A): {np.mean(A_values):.3f} ± {np.std(A_values):.3f}")
        print(f"Transition scale (k_c): {np.mean(k_c_values):.3f} ± {np.std(k_c_values):.3f}")
        print(f"Low-k power (α): {np.mean(alpha_values):.3f} ± {np.std(alpha_values):.3f}")
        print(f"High-k suppression (β): {np.mean(beta_values):.3f} ± {np.std(beta_values):.3f}")
        print(f"Damping parameter (δ): {np.mean(delta_values):.3f} ± {np.std(delta_values):.3f}")
        print(f"Damping scale (k_damp): {np.mean(k_damp_values):.3f} ± {np.std(k_damp_values):.3f}")
        print(f"Average R²: {np.mean(r2_values):.4f}")

if __name__ == "__main__":
    print("Testing Analytic Transfer Function Fits")
    print("=" * 50)
    
    # Run the test
    results = test_analytic_function_on_sample_data()
    
    if results:
        # Plot best fits
        plot_best_fits(results)
        
        # Analyze parameter trends
        analyze_parameter_trends(results)
        
        print(f"\nSuccessfully tested {len(results)} fits")
    else:
        print("No successful fits obtained")

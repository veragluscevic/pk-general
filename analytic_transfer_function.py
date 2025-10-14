#!/usr/bin/env python3
"""
Analytic transfer function for interacting dark matter scenarios.

This module provides an analytic function to describe the normalized transfer functions
T_IDM(k) / T_ΛCDM(k) for different interacting dark matter models.

The function captures the essential physics of:
- Free-streaming suppression at small scales
- Interaction-dependent transitions
- Mass-dependent cutoffs
- Cross-section dependent amplitudes

Based on the physics of interacting dark matter and structure formation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os
import re

def analytic_transfer_function(k, A, k_c, alpha, beta, gamma=2.0):
    """
    Analytic form for normalized transfer function T_IDM(k) / T_ΛCDM(k).
    
    This function captures the essential features of interacting dark matter
    transfer functions with minimal parameters.
    
    Parameters:
    -----------
    k : array_like
        Wavenumber in h/Mpc
    A : float
        Amplitude of deviation from ΛCDM
    k_c : float
        Characteristic wavenumber where transition occurs (h/Mpc)
    alpha : float
        Power-law index for low-k behavior
    beta : float
        Power-law index for high-k suppression
    gamma : float, optional
        Transition sharpness parameter (default: 2.0)
        
    Returns:
    --------
    T_norm : array_like
        Normalized transfer function T_IDM(k) / T_ΛCDM(k)
        
    Notes:
    ------
    The function has the form:
    T_norm(k) = 1 + A * (k/k_c)^alpha / (1 + (k/k_c)^gamma)^beta
    
    This captures:
    - Low-k: T_norm ≈ 1 + A * (k/k_c)^alpha (small deviations)
    - High-k: T_norm ≈ 1 + A * (k/k_c)^(alpha - gamma*beta) (suppression)
    - Transition at k ≈ k_c
    """
    
    # Avoid division by zero
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Main functional form
    ratio = k_safe / k_c
    numerator = ratio**alpha
    denominator = (1 + ratio**gamma)**beta
    
    T_norm = 1 + A * numerator / denominator
    
    return T_norm

def enhanced_analytic_function(k, A, k_c, alpha, beta, delta=0.0, k_damp=None):
    """
    Enhanced analytic form with additional physics.
    
    This includes:
    - Free-streaming cutoff
    - Interaction damping
    - Scale-dependent transitions
    
    Parameters:
    -----------
    k : array_like
        Wavenumber in h/Mpc
    A : float
        Amplitude parameter
    k_c : float
        Characteristic transition scale
    alpha : float
        Low-k power law index
    beta : float
        High-k suppression index
    delta : float
        Additional suppression parameter
    k_damp : float, optional
        Damping scale for free-streaming effects
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Base form
    ratio = k_safe / k_c
    numerator = ratio**alpha
    denominator = (1 + ratio**2)**beta
    
    T_base = 1 + A * numerator / denominator
    
    # Add damping term if specified
    if k_damp is not None:
        damping = np.exp(-(k_safe / k_damp)**2)
        T_base = T_base * damping + (1 - damping)
    
    # Additional suppression term
    if delta != 0:
        suppression = 1 / (1 + delta * (k_safe / k_c)**2)
        T_base = T_base * suppression + (1 - suppression)
    
    return T_base

def fit_analytic_function(k_data, T_data, function_type='basic', initial_guess=None):
    """
    Fit the analytic function to data.
    
    Parameters:
    -----------
    k_data : array_like
        Wavenumber data
    T_data : array_like
        Normalized transfer function data
    function_type : str
        'basic' or 'enhanced'
    initial_guess : array_like, optional
        Initial parameter guess
        
    Returns:
    --------
    popt : array
        Optimal parameters
    pcov : array
        Parameter covariance matrix
    """
    
    if function_type == 'basic':
        func = analytic_transfer_function
        if initial_guess is None:
            initial_guess = [0.1, 1.0, 1.0, 1.0]  # A, k_c, alpha, beta
    elif function_type == 'enhanced':
        func = enhanced_analytic_function
        if initial_guess is None:
            initial_guess = [0.1, 1.0, 1.0, 1.0, 0.0, 10.0]  # A, k_c, alpha, beta, delta, k_damp
    else:
        raise ValueError("function_type must be 'basic' or 'enhanced'")
    
    try:
        popt, pcov = curve_fit(func, k_data, T_data, p0=initial_guess, 
                              maxfev=10000, bounds=(0, [10, 100, 5, 5, 1, 100]))
        return popt, pcov
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None

def parameter_interpretation(params, function_type='basic'):
    """
    Interpret the fitted parameters in physical terms.
    
    Parameters:
    -----------
    params : array_like
        Fitted parameters
    function_type : str
        Type of function used
        
    Returns:
    --------
    interpretation : dict
        Physical interpretation of parameters
    """
    
    if function_type == 'basic':
        A, k_c, alpha, beta = params
        interpretation = {
            'amplitude': A,
            'transition_scale': k_c,
            'low_k_power': alpha,
            'high_k_suppression': beta,
            'physical_meaning': {
                'A': f'Amplitude of deviation from ΛCDM: {A:.3f}',
                'k_c': f'Transition scale: {k_c:.3f} h/Mpc',
                'alpha': f'Low-k power law index: {alpha:.3f}',
                'beta': f'High-k suppression index: {beta:.3f}'
            }
        }
    else:
        A, k_c, alpha, beta, delta, k_damp = params
        interpretation = {
            'amplitude': A,
            'transition_scale': k_c,
            'low_k_power': alpha,
            'high_k_suppression': beta,
            'damping_parameter': delta,
            'damping_scale': k_damp,
            'physical_meaning': {
                'A': f'Amplitude: {A:.3f}',
                'k_c': f'Transition scale: {k_c:.3f} h/Mpc',
                'alpha': f'Low-k power: {alpha:.3f}',
                'beta': f'High-k suppression: {beta:.3f}',
                'delta': f'Additional suppression: {delta:.3f}',
                'k_damp': f'Damping scale: {k_damp:.3f} h/Mpc'
            }
        }
    
    return interpretation

def plot_fit_comparison(k_data, T_data, params, function_type='basic', 
                       title="Analytic Function Fit", save_plot=True):
    """
    Plot the data and fitted analytic function for comparison.
    
    Parameters:
    -----------
    k_data : array_like
        Wavenumber data
    T_data : array_like
        Transfer function data
    params : array_like
        Fitted parameters
    function_type : str
        Type of function used
    title : str
        Plot title
    save_plot : bool
        Whether to save the plot
    """
    
    plt.figure(figsize=(10, 6))
    
    # Plot data
    plt.semilogx(k_data, T_data, 'o', alpha=0.6, markersize=3, label='Data')
    
    # Generate smooth curve for plotting
    k_smooth = np.logspace(np.log10(k_data.min()), np.log10(k_data.max()), 200)
    
    if function_type == 'basic':
        T_smooth = analytic_transfer_function(k_smooth, *params)
    else:
        T_smooth = enhanced_analytic_function(k_smooth, *params)
    
    plt.semilogx(k_smooth, T_smooth, 'r-', linewidth=2, label='Analytic Fit')
    
    # Add reference line
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        filename = f'analytic_fit_{function_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    plt.show()

if __name__ == "__main__":
    print("Analytic Transfer Function Module")
    print("This module provides functions to fit analytic forms to IDM transfer functions.")
    print("\nAvailable functions:")
    print("- analytic_transfer_function(): Basic 4-parameter form")
    print("- enhanced_analytic_function(): Enhanced 6-parameter form")
    print("- fit_analytic_function(): Fitting routine")
    print("- parameter_interpretation(): Physical interpretation")
    print("- plot_fit_comparison(): Visualization")

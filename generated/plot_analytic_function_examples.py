#!/usr/bin/env python3
"""
Example plots of the analytic transfer function.

This script demonstrates how the analytic function
$$\frac{T_{\mathrm{IDM}}(k)}{T_{\Lambda\mathrm{CDM}}(k)} = 1 + A \frac{(k/k_c)^n}{1 + (k/k_c)^m}$$

behaves with different parameter values.
"""

import numpy as np
import matplotlib.pyplot as plt

def analytic_transfer_function(k, A, k_c, n, m):
    """
    Analytic form for normalized transfer function T_IDM(k) / T_ΛCDM(k).
    
    Parameters:
    -----------
    k : array_like
        Wavenumber in h/Mpc
    A : float
        Amplitude of deviation from ΛCDM
    k_c : float
        Characteristic transition scale in h/Mpc
    n : float
        Low-k power law index
    m : float
        High-k suppression index
        
    Returns:
    --------
    T_norm : array_like
        Normalized transfer function T_IDM(k) / T_ΛCDM(k)
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    ratio = k_safe / k_c
    numerator = ratio**n
    denominator = 1 + ratio**m
    
    T_norm = 1 + A * numerator / denominator
    
    return T_norm

def plot_analytic_function_examples():
    """Create example plots showing different parameter effects."""
    
    # Create k range (focusing on k > 0.01 as in our data)
    k = np.logspace(-2, 1.5, 200)  # 0.01 to ~30 h/Mpc
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Analytic Transfer Function Examples\nT_IDM(k) / T_ΛCDM(k) = 1 + A × (k/k_c)^n / (1 + (k/k_c)^m)', 
                 fontsize=14, y=0.98)
    
    # Plot 1: Effect of amplitude A
    ax1 = axes[0, 0]
    A_values = [-0.5, -0.2, 0.2, 0.5]
    k_c, n, m = 1.0, 2.0, 2.0  # Fixed parameters
    
    for A in A_values:
        T = analytic_transfer_function(k, A, k_c, n, m)
        label = f'A = {A}'
        ax1.semilogx(k, T, label=label, linewidth=2)
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax1.set_title('Effect of Amplitude A\n(k_c=1.0, n=2.0, m=2.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Effect of transition scale k_c
    ax2 = axes[0, 1]
    k_c_values = [0.1, 0.5, 1.0, 2.0]
    A, n, m = 0.3, 2.0, 2.0  # Fixed parameters
    
    for k_c in k_c_values:
        T = analytic_transfer_function(k, A, k_c, n, m)
        label = f'k_c = {k_c}'
        ax2.semilogx(k, T, label=label, linewidth=2)
    
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax2.set_title('Effect of Transition Scale k_c\n(A=0.3, n=2.0, m=2.0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Effect of low-k power n
    ax3 = axes[1, 0]
    n_values = [1.0, 1.5, 2.0, 3.0]
    A, k_c, m = 0.3, 1.0, 2.0  # Fixed parameters
    
    for n in n_values:
        T = analytic_transfer_function(k, A, k_c, n, m)
        label = f'n = {n}'
        ax3.semilogx(k, T, label=label, linewidth=2)
    
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax3.set_title('Effect of Low-k Power n\n(A=0.3, k_c=1.0, m=2.0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effect of high-k suppression m
    ax4 = axes[1, 1]
    m_values = [1.0, 1.5, 2.0, 3.0]
    A, k_c, n = 0.3, 1.0, 2.0  # Fixed parameters
    
    for m in m_values:
        T = analytic_transfer_function(k, A, k_c, n, m)
        label = f'm = {m}'
        ax4.semilogx(k, T, label=label, linewidth=2)
    
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax4.set_xlabel('k [h/Mpc]')
    ax4.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax4.set_title('Effect of High-k Suppression m\n(A=0.3, k_c=1.0, n=2.0)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analytic_function_examples.png', dpi=300, bbox_inches='tight')
    print("Example plots saved as analytic_function_examples.png")
    plt.show()

def plot_realistic_idm_scenarios():
    """Plot realistic IDM scenarios based on our data analysis."""
    
    # Create k range
    k = np.logspace(-2, 1.5, 200)  # 0.01 to ~30 h/Mpc
    
    plt.figure(figsize=(10, 6))
    
    # Realistic parameter sets based on our analysis
    scenarios = [
        {'params': (0.1, 2.0, 2.5, 2.0), 'label': 'Weak IDM (np0)', 'color': 'blue'},
        {'params': (0.3, 1.5, 2.0, 2.5), 'label': 'Moderate IDM (np2)', 'color': 'red'},
        {'params': (0.5, 1.0, 1.8, 3.0), 'label': 'Strong IDM (np4)', 'color': 'green'},
        {'params': (-0.2, 3.0, 3.0, 2.0), 'label': 'Suppression (low mass)', 'color': 'orange'},
        {'params': (0.8, 0.5, 1.5, 4.0), 'label': 'Enhancement (high cross-section)', 'color': 'purple'}
    ]
    
    for scenario in scenarios:
        A, k_c, n, m = scenario['params']
        T = analytic_transfer_function(k, A, k_c, n, m)
        plt.semilogx(k, T, label=scenario['label'], color=scenario['color'], 
                    linewidth=2.5, alpha=0.8)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='ΛCDM')
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title('Realistic IDM Transfer Function Scenarios\nAnalytic Function with Different Parameter Sets', 
              fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text box with function
    function_text = 'T_IDM(k) / T_ΛCDM(k) = 1 + A × (k/k_c)^n / (1 + (k/k_c)^m)'
    plt.text(0.02, 0.98, function_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('realistic_idm_scenarios.png', dpi=300, bbox_inches='tight')
    print("Realistic scenarios saved as realistic_idm_scenarios.png")
    plt.show()

def plot_parameter_space_exploration():
    """Explore the parameter space systematically."""
    
    # Create k range
    k = np.logspace(-2, 1.5, 200)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Parameter Space Exploration', fontsize=14)
    
    # Different interaction types (np0, np2, np4)
    interaction_types = [
        {'params': (0.05, 3.0, 2.0, 2.0), 'label': 'np0 (weak interaction)', 'color': 'blue'},
        {'params': (0.2, 1.5, 2.2, 2.5), 'label': 'np2 (moderate interaction)', 'color': 'red'},
        {'params': (0.4, 1.0, 1.8, 3.0), 'label': 'np4 (strong interaction)', 'color': 'green'}
    ]
    
    ax1 = axes[0]
    for interaction in interaction_types:
        A, k_c, n, m = interaction['params']
        T = analytic_transfer_function(k, A, k_c, n, m)
        ax1.semilogx(k, T, label=interaction['label'], color=interaction['color'], linewidth=2)
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax1.set_title('Interaction Type Effects')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Different masses
    mass_scenarios = [
        {'params': (-0.1, 5.0, 3.0, 2.0), 'label': 'Low mass (10⁻⁵ GeV)', 'color': 'cyan'},
        {'params': (0.1, 2.0, 2.5, 2.5), 'label': 'Medium mass (10⁻² GeV)', 'color': 'orange'},
        {'params': (0.3, 0.8, 2.0, 3.0), 'label': 'High mass (1 GeV)', 'color': 'magenta'}
    ]
    
    ax2 = axes[1]
    for mass in mass_scenarios:
        A, k_c, n, m = mass['params']
        T = analytic_transfer_function(k, A, k_c, n, m)
        ax2.semilogx(k, T, label=mass['label'], color=mass['color'], linewidth=2)
    
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax2.set_title('Mass Effects')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Different cross-sections
    cross_section_scenarios = [
        {'params': (0.05, 2.0, 2.0, 2.0), 'label': 'Low σ (10⁻³⁰ cm²)', 'color': 'lightblue'},
        {'params': (0.2, 1.5, 2.2, 2.5), 'label': 'Medium σ (10⁻²⁷ cm²)', 'color': 'lightgreen'},
        {'params': (0.6, 1.0, 1.8, 3.5), 'label': 'High σ (10⁻²⁵ cm²)', 'color': 'lightcoral'}
    ]
    
    ax3 = axes[2]
    for cross_section in cross_section_scenarios:
        A, k_c, n, m = cross_section['params']
        T = analytic_transfer_function(k, A, k_c, n, m)
        ax3.semilogx(k, T, label=cross_section['label'], color=cross_section['color'], linewidth=2)
    
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax3.set_title('Cross-Section Effects')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_space_exploration.png', dpi=300, bbox_inches='tight')
    print("Parameter space exploration saved as parameter_space_exploration.png")
    plt.show()

if __name__ == "__main__":
    print("Creating example plots of the analytic transfer function...")
    print("Function: T_IDM(k) / T_ΛCDM(k) = 1 + A × (k/k_c)^n / (1 + (k/k_c)^m)")
    print()
    
    # Create all example plots
    plot_analytic_function_examples()
    plot_realistic_idm_scenarios()
    plot_parameter_space_exploration()
    
    print("\nAll example plots created successfully!")
    print("The analytic function demonstrates:")
    print("- A: Controls amplitude of deviation from ΛCDM")
    print("- k_c: Sets the transition scale")
    print("- n: Controls low-k behavior")
    print("- m: Controls high-k suppression")

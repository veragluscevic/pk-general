#!/usr/bin/env python3
"""
Enhanced oscillatory transfer function with exponential base and k-dependent parameters.

$$\frac{T_{\mathrm{IDM}}(k)}{T_{\Lambda\mathrm{CDM}}(k)} = 1 + A \exp\left[-\left(\frac{k}{k_c}\right)^{\alpha}\right] + A_{\mathrm{osc}}(k) \cos\left(\frac{2\pi k}{k_{\mathrm{osc}}(k)} + \phi(k)\right)$$

Where:
- $A_{\mathrm{osc}}(k) = A_0 \left(\frac{k}{k_{\mathrm{ref}}}\right)^{\beta} \exp\left(-\frac{k}{k_{\mathrm{decay}}}\right)$
- $k_{\mathrm{osc}}(k) = k_0 \left(\frac{k}{k_{\mathrm{ref}}}\right)^{\gamma}$
- $\phi(k) = \phi_0 + \phi_1 \left(\frac{k}{k_{\mathrm{ref}}}\right)^{\delta}$
"""

import numpy as np
import matplotlib.pyplot as plt

def enhanced_oscillatory_function(k, A, k_c, alpha, A0, beta, k_decay, k0, gamma, phi0, delta, k_ref=1.0, phi1=0.0):
    """
    Enhanced oscillatory transfer function with k-dependent parameters.
    
    Parameters:
    -----------
    k : array_like
        Wavenumber in h/Mpc
    A : float
        Amplitude of smooth deviation
    k_c : float
        Characteristic scale for exponential
    alpha : float
        Exponential power index
    A0 : float
        Base oscillation amplitude
    beta : float
        k-dependence of oscillation amplitude
    k_decay : float
        High-k damping scale for oscillations
    k0 : float
        Base oscillation wavelength
    gamma : float
        k-dependence of oscillation frequency
    phi0 : float
        Base phase offset
    delta : float
        k-dependence of phase
    k_ref : float, optional
        Reference scale (default: 1.0)
    phi1 : float, optional
        Phase evolution amplitude (default: 0.0)
        
    Returns:
    --------
    T_norm : array_like
        Enhanced normalized transfer function
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Base exponential function
    T_base = 1 + A * np.exp(-(k_safe / k_c)**alpha)
    
    # k-dependent oscillation amplitude
    A_osc_k = A0 * (k_safe / k_ref)**beta * np.exp(-k_safe / k_decay)
    
    # k-dependent oscillation frequency
    k_osc_k = k0 * (k_safe / k_ref)**gamma
    
    # k-dependent phase
    phi_k = phi0 + phi1 * (k_safe / k_ref)**delta
    
    # Oscillatory component
    oscillation = A_osc_k * np.cos(2 * np.pi * k_safe / k_osc_k + phi_k)
    
    T_norm = T_base + oscillation
    
    return T_norm

def plot_base_exponential_function():
    """Plot the exponential base function behavior."""
    
    k = np.logspace(-2, 1.5, 200)
    
    plt.figure(figsize=(12, 8))
    
    # Effect of A (amplitude)
    plt.subplot(2, 2, 1)
    A_values = [-0.5, -0.2, 0.2, 0.5]
    k_c, alpha = 1.0, 2.0
    
    for A in A_values:
        T = 1 + A * np.exp(-(k / k_c)**alpha)
        plt.semilogx(k, T, label=f'A = {A}', linewidth=2)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_base(k)')
    plt.title('Effect of Amplitude A\n(k_c=1.0, α=2.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Effect of k_c (characteristic scale)
    plt.subplot(2, 2, 2)
    k_c_values = [0.5, 1.0, 2.0, 4.0]
    A, alpha = 0.3, 2.0
    
    for k_c in k_c_values:
        T = 1 + A * np.exp(-(k / k_c)**alpha)
        plt.semilogx(k, T, label=f'k_c = {k_c}', linewidth=2)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_base(k)')
    plt.title('Effect of Characteristic Scale k_c\n(A=0.3, α=2.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Effect of alpha (exponential power)
    plt.subplot(2, 2, 3)
    alpha_values = [1.0, 1.5, 2.0, 3.0]
    A, k_c = 0.3, 1.0
    
    for alpha in alpha_values:
        T = 1 + A * np.exp(-(k / k_c)**alpha)
        plt.semilogx(k, T, label=f'α = {alpha}', linewidth=2)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_base(k)')
    plt.title('Effect of Exponential Power α\n(A=0.3, k_c=1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Combined effect
    plt.subplot(2, 2, 4)
    scenarios = [
        {'params': (0.3, 1.0, 2.0), 'label': 'Enhancement', 'color': 'blue'},
        {'params': (-0.3, 1.0, 2.0), 'label': 'Suppression', 'color': 'red'},
        {'params': (0.2, 2.0, 1.5), 'label': 'Wide enhancement', 'color': 'green'},
        {'params': (-0.4, 0.5, 3.0), 'label': 'Sharp suppression', 'color': 'orange'},
    ]
    
    for scenario in scenarios:
        A, k_c, alpha = scenario['params']
        T = 1 + A * np.exp(-(k / k_c)**alpha)
        plt.semilogx(k, T, label=scenario['label'], color=scenario['color'], linewidth=2)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('T_base(k)')
    plt.title('Combined Effects')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exponential_base_function.png', dpi=300, bbox_inches='tight')
    print("Exponential base function saved as exponential_base_function.png")
    plt.show()

def plot_k_dependent_oscillations():
    """Plot k-dependent oscillation parameters."""
    
    k = np.logspace(-2, 1.5, 400)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('k-Dependent Oscillation Parameters', fontsize=14)
    
    # k-dependent amplitude
    ax1 = axes[0, 0]
    beta_values = [-0.5, 0.0, 0.5, 1.0]
    A0, k_decay = 0.2, 2.0
    k_ref = 1.0
    
    for beta in beta_values:
        A_osc_k = A0 * (k / k_ref)**beta * np.exp(-k / k_decay)
        ax1.semilogx(k, A_osc_k, label=f'β = {beta}', linewidth=2)
    
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('A_osc(k)')
    ax1.set_title('k-Dependent Oscillation Amplitude\n(A₀=0.2, k_decay=2.0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # k-dependent frequency
    ax2 = axes[0, 1]
    gamma_values = [-0.5, 0.0, 0.5, 1.0]
    k0 = 0.5
    
    for gamma in gamma_values:
        k_osc_k = k0 * (k / k_ref)**gamma
        ax2.semilogx(k, k_osc_k, label=f'γ = {gamma}', linewidth=2)
    
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('k_osc(k)')
    ax2.set_title('k-Dependent Oscillation Frequency\n(k₀=0.5)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # k-dependent phase
    ax3 = axes[1, 0]
    delta_values = [0.0, 0.5, 1.0, 1.5]
    phi0, phi1 = 0.0, 1.0
    
    for delta in delta_values:
        phi_k = phi0 + phi1 * (k / k_ref)**delta
        ax3.semilogx(k, phi_k, label=f'δ = {delta}', linewidth=2)
    
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('φ(k)')
    ax3.set_title('k-Dependent Phase\n(φ₀=0.0, φ₁=1.0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Combined oscillation effect
    ax4 = axes[1, 1]
    scenarios = [
        {'params': (0.1, 0.0, 3.0, 0.4, 0.0, 0.0, 0.0), 'label': 'Constant oscillations', 'color': 'blue'},
        {'params': (0.15, 0.5, 2.0, 0.4, 0.5, 0.0, 0.5), 'label': 'Growing amplitude', 'color': 'red'},
        {'params': (0.1, -0.5, 1.5, 0.3, 1.0, 0.0, 0.0), 'label': 'Decreasing frequency', 'color': 'green'},
        {'params': (0.12, 0.3, 2.5, 0.35, 0.3, 1.0, 0.8), 'label': 'Complex evolution', 'color': 'purple'},
    ]
    
    for scenario in scenarios:
        A0, beta, k_decay, k0, gamma, phi0, delta = scenario['params']
        A, k_c, alpha = 0.2, 1.5, 2.0  # Base function parameters
        T = enhanced_oscillatory_function(k, A, k_c, alpha, A0, beta, k_decay, 
                                        k0, gamma, phi0, delta)
        ax4.semilogx(k, T, label=scenario['label'], color=scenario['color'], linewidth=2)
    
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax4.set_xlabel('k [h/Mpc]')
    ax4.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax4.set_title('Combined k-Dependent Oscillations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('k_dependent_oscillations.png', dpi=300, bbox_inches='tight')
    print("k-dependent oscillations saved as k_dependent_oscillations.png")
    plt.show()

def plot_enhanced_function_scenarios():
    """Plot realistic enhanced function scenarios."""
    
    k = np.logspace(-2, 1.5, 500)
    
    plt.figure(figsize=(14, 8))
    
    # Realistic IDM scenarios with enhanced function
    scenarios = [
        # Weak IDM with mild oscillations
        {'params': (0.1, 2.0, 2.0, 0.08, 0.0, 3.0, 0.4, 0.0, 0.0, 0.0), 
         'label': 'Weak IDM (np0)', 'color': 'blue', 'linestyle': '-'},
        
        # Moderate IDM with growing oscillations
        {'params': (0.2, 1.5, 2.2, 0.12, 0.3, 2.5, 0.35, 0.2, 0.0, 0.0), 
         'label': 'Moderate IDM (np2)', 'color': 'red', 'linestyle': '-'},
        
        # Strong IDM with complex oscillations
        {'params': (0.3, 1.0, 1.8, 0.15, 0.5, 2.0, 0.3, 0.5, 0.5, 0.5), 
         'label': 'Strong IDM (np4)', 'color': 'green', 'linestyle': '-'},
        
        # Suppression with decreasing oscillations
        {'params': (-0.2, 2.5, 2.5, 0.1, -0.3, 3.0, 0.5, -0.2, 0.0, 0.0), 
         'label': 'Suppression scenario', 'color': 'orange', 'linestyle': '-'},
        
        # Enhancement with high-frequency oscillations
        {'params': (0.4, 0.8, 1.5, 0.2, 0.2, 1.5, 0.2, 1.0, 0.0, 0.0), 
         'label': 'High-frequency enhancement', 'color': 'purple', 'linestyle': '-'},
    ]
    
    for scenario in scenarios:
        params = scenario['params']
        A, k_c, alpha, A0, beta, k_decay, k0, gamma, phi0, delta = params
        T = enhanced_oscillatory_function(k, A, k_c, alpha, A0, beta, k_decay, 
                                        k0, gamma, phi0, delta)
        plt.semilogx(k, T, label=scenario['label'], color=scenario['color'], 
                    linewidth=2.5, alpha=0.8, linestyle=scenario['linestyle'])
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM')
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title('Enhanced Oscillatory Transfer Function Scenarios\nExponential Base + k-Dependent Oscillations', 
              fontsize=14, pad=20)
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    
    # Add function description
    function_text = ('T_IDM(k) / T_ΛCDM(k) = 1 + A × exp(-(k/k_c)^α) +\n'
                    'A₀ × (k/k_ref)^β × exp(-k/k_decay) × cos(2πk/k₀×(k/k_ref)^γ + φ₀ + φ₁×(k/k_ref)^δ)')
    plt.text(0.02, 0.98, function_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('enhanced_function_scenarios.png', dpi=300, bbox_inches='tight')
    print("Enhanced function scenarios saved as enhanced_function_scenarios.png")
    plt.show()

def compare_function_forms():
    """Compare the original, oscillatory, and enhanced functions."""
    
    k = np.logspace(-2, 1.5, 400)
    
    plt.figure(figsize=(12, 8))
    
    # Original function (smooth only)
    A, k_c, n, m = 0.2, 1.5, 2.0, 2.5
    ratio = k / k_c
    T_original = 1 + A * (ratio**n) / (1 + ratio**m)
    plt.semilogx(k, T_original, label='Original (smooth)', color='gray', 
                linewidth=2, linestyle='--', alpha=0.7)
    
    # Simple oscillatory function
    T_osc = enhanced_oscillatory_function(k, 0.2, 1.5, 2.0, 0.1, 0.0, 3.0, 0.4, 0.0, 0.0, 0.0)
    plt.semilogx(k, T_osc, label='Simple oscillatory', color='blue', linewidth=2)
    
    # Enhanced function with k-dependent parameters
    T_enhanced = enhanced_oscillatory_function(k, 0.2, 1.5, 2.0, 0.12, 0.3, 2.5, 0.35, 0.2, 0.5, 0.5)
    plt.semilogx(k, T_enhanced, label='Enhanced (k-dependent)', color='red', linewidth=2.5)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM')
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title('Comparison of Function Forms\nFrom Simple to Enhanced Oscillatory', 
              fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('function_form_comparison.png', dpi=300, bbox_inches='tight')
    print("Function comparison saved as function_form_comparison.png")
    plt.show()

if __name__ == "__main__":
    print("Creating enhanced oscillatory transfer function plots...")
    print("Function: T_IDM(k) / T_ΛCDM(k) = 1 + A × exp(-(k/k_c)^α) + A_osc(k) × cos(2πk/k_osc(k) + φ(k))")
    print("With k-dependent oscillation parameters")
    print()
    
    # Create all plots
    plot_base_exponential_function()
    plot_k_dependent_oscillations()
    plot_enhanced_function_scenarios()
    compare_function_forms()
    
    print("\nAll enhanced function plots created successfully!")
    print("Key features:")
    print("- Exponential base function for smoother behavior")
    print("- k-dependent oscillation amplitude: A_osc(k)")
    print("- k-dependent oscillation frequency: k_osc(k)")
    print("- k-dependent phase evolution: φ(k)")
    print("- Total of 10 parameters for maximum flexibility")

#!/usr/bin/env python3
"""
Enhanced analytic transfer function with oscillatory features.

This module provides an analytic function that can capture oscillatory
features in IDM transfer functions, which are important for accurate
representation of the physics.

The enhanced function includes:
$$\frac{T_{\mathrm{IDM}}(k)}{T_{\Lambda\mathrm{CDM}}(k)} = 1 + A \exp\left[-\left(\frac{k}{k_c}\right)^m\right] + A_{\mathrm{osc}}(k) \cos\left(\frac{2\pi k}{k_{\mathrm{osc}}(k)} + \phi\right) \exp\left(-\frac{k}{k_{\mathrm{damp}}}\right)$$

Where:
- $A_{\mathrm{osc}}(k) = A_0 \left(\frac{k}{k_{\mathrm{ref}}}\right)^{\alpha} \exp\left[-\left(\frac{k}{k_{\mathrm{damp},A}}\right)^{\beta}\right]$
- $k_{\mathrm{osc}}(k) = k_0 \left[1 + \gamma \ln\left(\frac{k}{k_{\mathrm{ref}}}\right)\right]$
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def oscillatory_transfer_function(k, A, k_c, n, m, A_osc, k_osc, phase, decay=1.0):
    """
    Enhanced analytic form with oscillatory features.
    
    T_IDM(k) / T_ΛCDM(k) = 1 + A × (k/k_c)^n / (1 + (k/k_c)^m) + A_osc × cos(2πk/k_osc + phase) × exp(-k/decay)
    
    Parameters:
    -----------
    k : array_like
        Wavenumber in h/Mpc
    A : float
        Amplitude of smooth deviation from ΛCDM
    k_c : float
        Characteristic transition scale in h/Mpc
    n : float
        Low-k power law index
    m : float
        High-k suppression index
    A_osc : float
        Amplitude of oscillatory component
    k_osc : float
        Oscillation wavelength scale in h/Mpc
    phase : float
        Phase offset of oscillations
    decay : float, optional
        Decay scale for oscillations (default: 1.0)
        
    Returns:
    --------
    T_norm : array_like
        Normalized transfer function with oscillations
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Base smooth component
    ratio = k_safe / k_c
    numerator = ratio**n
    denominator = 1 + ratio**m
    T_smooth = 1 + A * numerator / denominator
    
    # Oscillatory component
    oscillation = A_osc * np.cos(2 * np.pi * k_safe / k_osc + phase) * np.exp(-k_safe / decay)
    
    T_norm = T_smooth + oscillation
    
    return T_norm

def damped_oscillatory_function(k, A, k_c, n, m, A_osc, k_osc, phase, damping_rate=0.5):
    """
    Alternative oscillatory form with power-law damping.
    
    T_IDM(k) / T_ΛCDM(k) = 1 + A × (k/k_c)^n / (1 + (k/k_c)^m) + A_osc × cos(2πk/k_osc + phase) × (k/k_c)^(-damping_rate)
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Base smooth component
    ratio = k_safe / k_c
    numerator = ratio**n
    denominator = 1 + ratio**m
    T_smooth = 1 + A * numerator / denominator
    
    # Oscillatory component with power-law damping
    damping = ratio**(-damping_rate)
    oscillation = A_osc * np.cos(2 * np.pi * k_safe / k_osc + phase) * damping
    
    T_norm = T_smooth + oscillation
    
    return T_norm

def multi_frequency_oscillatory_function(k, A, k_c, n, m, A_osc1, k_osc1, phase1, A_osc2, k_osc2, phase2):
    """
    Oscillatory function with two frequency components.
    
    This can capture multiple oscillatory features that might arise from
    different physical processes in IDM scenarios.
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Base smooth component
    ratio = k_safe / k_c
    numerator = ratio**n
    denominator = 1 + ratio**m
    T_smooth = 1 + A * numerator / denominator
    
    # Two oscillatory components
    osc1 = A_osc1 * np.cos(2 * np.pi * k_safe / k_osc1 + phase1)
    osc2 = A_osc2 * np.cos(2 * np.pi * k_safe / k_osc2 + phase2)
    
    # Add exponential decay to oscillations
    decay_factor = np.exp(-k_safe / (2 * k_c))
    
    T_norm = T_smooth + (osc1 + osc2) * decay_factor
    
    return T_norm

def plot_oscillatory_examples():
    """Create example plots showing oscillatory features."""
    
    # Create k range
    k = np.logspace(-2, 1.5, 300)  # More points for smooth oscillations
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Oscillatory Transfer Function Examples\nT_IDM(k) / T_ΛCDM(k) with Oscillatory Features', 
                 fontsize=14, y=0.98)
    
    # Plot 1: Basic oscillatory function
    ax1 = axes[0, 0]
    scenarios = [
        {'params': (0.2, 1.5, 2.0, 2.5, 0.1, 0.5, 0, 2.0), 'label': 'Mild oscillations', 'color': 'blue'},
        {'params': (0.2, 1.5, 2.0, 2.5, 0.3, 0.3, np.pi/4, 1.5), 'label': 'Strong oscillations', 'color': 'red'},
        {'params': (0.2, 1.5, 2.0, 2.5, 0.05, 1.0, 0, 3.0), 'label': 'Long-wavelength oscillations', 'color': 'green'},
    ]
    
    for scenario in scenarios:
        T = oscillatory_transfer_function(k, *scenario['params'])
        ax1.semilogx(k, T, label=scenario['label'], color=scenario['color'], linewidth=2)
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax1.set_title('Basic Oscillatory Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Damped oscillations
    ax2 = axes[0, 1]
    scenarios = [
        {'params': (0.2, 1.5, 2.0, 2.5, 0.2, 0.4, 0, 0.3), 'label': 'Fast damping', 'color': 'purple'},
        {'params': (0.2, 1.5, 2.0, 2.5, 0.2, 0.4, 0, 1.0), 'label': 'Moderate damping', 'color': 'orange'},
        {'params': (0.2, 1.5, 2.0, 2.5, 0.2, 0.4, 0, 3.0), 'label': 'Slow damping', 'color': 'cyan'},
    ]
    
    for scenario in scenarios:
        T = oscillatory_transfer_function(k, *scenario['params'])
        ax2.semilogx(k, T, label=scenario['label'], color=scenario['color'], linewidth=2)
    
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax2.set_title('Damped Oscillations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Multi-frequency oscillations
    ax3 = axes[1, 0]
    scenarios = [
        {'params': (0.1, 1.5, 2.0, 2.5, 0.1, 0.5, 0, 0.2, 0.3, np.pi/2), 'label': 'Two frequencies', 'color': 'red'},
        {'params': (0.15, 1.2, 2.2, 2.3, 0.08, 0.4, np.pi/4, 0.15, 0.8, 0), 'label': 'Mixed frequencies', 'color': 'blue'},
    ]
    
    for scenario in scenarios:
        T = multi_frequency_oscillatory_function(k, *scenario['params'])
        ax3.semilogx(k, T, label=scenario['label'], color=scenario['color'], linewidth=2)
    
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax3.set_title('Multi-Frequency Oscillations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison with smooth function
    ax4 = axes[1, 1]
    
    # Smooth function
    T_smooth = oscillatory_transfer_function(k, 0.2, 1.5, 2.0, 2.5, 0, 0.5, 0, 2.0)
    ax4.semilogx(k, T_smooth, label='Smooth only', color='gray', linewidth=2, linestyle='--')
    
    # With oscillations
    T_osc = oscillatory_transfer_function(k, 0.2, 1.5, 2.0, 2.5, 0.15, 0.4, 0, 2.0)
    ax4.semilogx(k, T_osc, label='With oscillations', color='red', linewidth=2)
    
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax4.set_xlabel('k [h/Mpc]')
    ax4.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax4.set_title('Smooth vs Oscillatory')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oscillatory_function_examples.png', dpi=300, bbox_inches='tight')
    print("Oscillatory examples saved as oscillatory_function_examples.png")
    plt.show()

def plot_oscillation_parameters():
    """Show the effect of different oscillation parameters."""
    
    # Create k range
    k = np.logspace(-2, 1.5, 400)  # High resolution for oscillations
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Effect of Oscillation Parameters', fontsize=14)
    
    # Effect of oscillation amplitude A_osc
    ax1 = axes[0]
    A_osc_values = [0.05, 0.1, 0.2, 0.3]
    base_params = (0.2, 1.5, 2.0, 2.5, None, 0.4, 0, 2.0)  # A_osc will be filled
    
    for A_osc in A_osc_values:
        params = list(base_params)
        params[4] = A_osc  # Insert A_osc
        T = oscillatory_transfer_function(k, *params)
        ax1.semilogx(k, T, label=f'A_osc = {A_osc}', linewidth=2)
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax1.set_title('Oscillation Amplitude A_osc')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Effect of oscillation wavelength k_osc
    ax2 = axes[1]
    k_osc_values = [0.2, 0.4, 0.6, 1.0]
    base_params = (0.2, 1.5, 2.0, 2.5, 0.15, None, 0, 2.0)  # k_osc will be filled
    
    for k_osc in k_osc_values:
        params = list(base_params)
        params[5] = k_osc  # Insert k_osc
        T = oscillatory_transfer_function(k, *params)
        ax2.semilogx(k, T, label=f'k_osc = {k_osc}', linewidth=2)
    
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax2.set_title('Oscillation Wavelength k_osc')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Effect of phase
    ax3 = axes[2]
    phase_values = [0, np.pi/4, np.pi/2, np.pi]
    base_params = (0.2, 1.5, 2.0, 2.5, 0.15, 0.4, None, 2.0)  # phase will be filled
    
    for phase in phase_values:
        params = list(base_params)
        params[6] = phase  # Insert phase
        T = oscillatory_transfer_function(k, *params)
        ax3.semilogx(k, T, label=f'phase = {phase:.2f}', linewidth=2)
    
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('T_IDM(k) / T_ΛCDM(k)')
    ax3.set_title('Oscillation Phase')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oscillation_parameters.png', dpi=300, bbox_inches='tight')
    print("Oscillation parameters saved as oscillation_parameters.png")
    plt.show()

def create_enhanced_function_summary():
    """Create a summary of the enhanced oscillatory function."""
    
    print("\n" + "="*70)
    print("ENHANCED OSCILLATORY TRANSFER FUNCTION")
    print("="*70)
    
    print("\nBasic Form:")
    print("T_IDM(k) / T_ΛCDM(k) = 1 + A × (k/k_c)^n / (1 + (k/k_c)^m) + A_osc × cos(2πk/k_osc + phase) × exp(-k/decay)")
    
    print("\nParameters:")
    print("- A: Amplitude of smooth deviation")
    print("- k_c: Characteristic transition scale")
    print("- n: Low-k power law index")
    print("- m: High-k suppression index")
    print("- A_osc: Amplitude of oscillatory component")
    print("- k_osc: Oscillation wavelength scale")
    print("- phase: Phase offset of oscillations")
    print("- decay: Decay scale for oscillations")
    
    print("\nAlternative Forms Available:")
    print("1. Damped oscillatory: Power-law damping instead of exponential")
    print("2. Multi-frequency: Two oscillatory components")
    print("3. Custom combinations for specific physics")
    
    print("\nPhysical Interpretation:")
    print("- Oscillations can arise from:")
    print("  • Acoustic oscillations in the early universe")
    print("  • Quantum interference effects in FDM")
    print("  • Resonant interactions in IDM")
    print("  • Baryon-DM interactions")
    print("  • Neutrino mass effects")

if __name__ == "__main__":
    print("Creating oscillatory transfer function examples...")
    print("Enhanced function with oscillatory features for IDM physics")
    print()
    
    # Create example plots
    plot_oscillatory_examples()
    plot_oscillation_parameters()
    
    # Create summary
    create_enhanced_function_summary()
    
    print("\nAll oscillatory examples created successfully!")
    print("The enhanced function can now capture both smooth deviations and oscillatory features.")

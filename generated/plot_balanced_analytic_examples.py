#!/usr/bin/env python3
"""
Balanced examples of the analytic transfer function showing both enhancement and suppression.

This script demonstrates the analytic function with both positive and negative A values
to show the full range of IDM physics scenarios.

Function: $$\frac{T_{\mathrm{IDM}}(k)}{T_{\Lambda\mathrm{CDM}}(k)} = 1 + A \frac{(k/k_c)^n}{1 + (k/k_c)^m}$$
"""

import numpy as np
import matplotlib.pyplot as plt

def analytic_transfer_function(k, A, k_c, n, m):
    """
    Analytic form for normalized transfer function T_IDM(k) / T_ΛCDM(k).
    """
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    ratio = k_safe / k_c
    numerator = ratio**n
    denominator = 1 + ratio**m
    
    T_norm = 1 + A * numerator / denominator
    
    return T_norm

def plot_balanced_scenarios():
    """Create balanced plots showing both enhancement and suppression."""
    
    # Create k range
    k = np.logspace(-2, 1.5, 200)  # 0.01 to ~30 h/Mpc
    
    plt.figure(figsize=(12, 8))
    
    # Balanced scenarios showing both enhancement and suppression
    scenarios = [
        # Suppression scenarios (A < 0)
        {'params': (-0.3, 2.0, 2.5, 2.0), 'label': 'Suppression: A=-0.3', 'color': 'red', 'linestyle': '-'},
        {'params': (-0.1, 1.5, 2.0, 2.5), 'label': 'Mild Suppression: A=-0.1', 'color': 'orange', 'linestyle': '-'},
        
        # Enhancement scenarios (A > 0)
        {'params': (0.1, 1.5, 2.0, 2.5), 'label': 'Mild Enhancement: A=+0.1', 'color': 'lightblue', 'linestyle': '-'},
        {'params': (0.3, 2.0, 2.5, 2.0), 'label': 'Enhancement: A=+0.3', 'color': 'blue', 'linestyle': '-'},
        {'params': (0.5, 1.0, 1.8, 3.0), 'label': 'Strong Enhancement: A=+0.5', 'color': 'darkblue', 'linestyle': '-'},
        
        # Special cases
        {'params': (0.0, 1.0, 2.0, 2.0), 'label': 'No Deviation: A=0', 'color': 'gray', 'linestyle': '--'},
    ]
    
    for scenario in scenarios:
        A, k_c, n, m = scenario['params']
        T = analytic_transfer_function(k, A, k_c, n, m)
        plt.semilogx(k, T, label=scenario['label'], color=scenario['color'], 
                    linewidth=2.5, alpha=0.8, linestyle=scenario['linestyle'])
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM')
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title('Balanced IDM Transfer Function Examples\nShowing Both Enhancement (A>0) and Suppression (A<0)', 
              fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text box with function
    function_text = 'T_IDM(k) / T_ΛCDM(k) = 1 + A × (k/k_c)^n / (1 + (k/k_c)^m)'
    plt.text(0.02, 0.98, function_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    # Add interpretation text
    interpretation = 'A > 0: Enhancement relative to ΛCDM\nA < 0: Suppression relative to ΛCDM\nA = 0: No deviation (pure ΛCDM)'
    plt.text(0.98, 0.02, interpretation, transform=plt.gca().transAxes, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    plt.savefig('balanced_analytic_examples.png', dpi=300, bbox_inches='tight')
    print("Balanced examples saved as balanced_analytic_examples.png")
    plt.show()

def plot_amplitude_sweep():
    """Show the effect of sweeping A from negative to positive values."""
    
    # Create k range
    k = np.logspace(-2, 1.5, 200)
    
    plt.figure(figsize=(10, 6))
    
    # Sweep A from negative to positive
    A_values = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    k_c, n, m = 1.5, 2.2, 2.5  # Fixed parameters
    
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(A_values)))
    
    for i, A in enumerate(A_values):
        T = analytic_transfer_function(k, A, k_c, n, m)
        label = f'A = {A:+.1f}'
        plt.semilogx(k, T, label=label, color=colors[i], linewidth=2.5, alpha=0.8)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM')
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title('Amplitude Sweep: Effect of Parameter A\n(k_c=1.5, n=2.2, m=2.5)', fontsize=14, pad=20)
    plt.legend(fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=-0.5, vmax=0.5))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Amplitude A', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('amplitude_sweep.png', dpi=300, bbox_inches='tight')
    print("Amplitude sweep saved as amplitude_sweep.png")
    plt.show()

def plot_physical_scenarios():
    """Plot physically motivated scenarios for different IDM types."""
    
    # Create k range
    k = np.logspace(-2, 1.5, 200)
    
    plt.figure(figsize=(12, 8))
    
    # Physically motivated scenarios
    scenarios = [
        # Warm Dark Matter (WDM) - typically suppression
        {'params': (-0.4, 3.0, 2.5, 2.0), 'label': 'WDM: Free-streaming suppression', 'color': 'purple'},
        
        # Fuzzy Dark Matter (FDM) - can show both
        {'params': (-0.2, 2.0, 2.0, 2.5), 'label': 'FDM: Quantum pressure effects', 'color': 'cyan'},
        
        # Interacting Dark Matter - enhancement possible
        {'params': (0.2, 1.5, 2.2, 2.5), 'label': 'IDM: Interaction enhancement', 'color': 'green'},
        {'params': (-0.3, 1.0, 2.5, 3.0), 'label': 'IDM: Interaction suppression', 'color': 'orange'},
        
        # Millicharged DM - typically enhancement
        {'params': (0.4, 1.2, 1.8, 2.8), 'label': 'Millicharged DM: EM coupling', 'color': 'red'},
        
        # Neutrino mass effects - mild suppression
        {'params': (-0.1, 2.5, 2.2, 2.2), 'label': 'Neutrino mass: Mild suppression', 'color': 'brown'},
    ]
    
    for scenario in scenarios:
        A, k_c, n, m = scenario['params']
        T = analytic_transfer_function(k, A, k_c, n, m)
        plt.semilogx(k, T, label=scenario['label'], color=scenario['color'], 
                    linewidth=2.5, alpha=0.8)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2, label='ΛCDM')
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title('Physical IDM Scenarios\nDifferent Dark Matter Physics with Both Enhancement and Suppression', 
              fontsize=14, pad=20)
    plt.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    
    # Add interpretation
    interpretation = ('A > 0: Enhancement (e.g., millicharged DM, some IDM)\n'
                     'A < 0: Suppression (e.g., WDM, FDM, neutrino mass)\n'
                     'Magnitude: Depends on interaction strength')
    plt.text(0.02, 0.98, interpretation, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    plt.savefig('physical_idm_scenarios.png', dpi=300, bbox_inches='tight')
    print("Physical scenarios saved as physical_idm_scenarios.png")
    plt.show()

if __name__ == "__main__":
    print("Creating balanced examples of the analytic transfer function...")
    print("Function: T_IDM(k) / T_ΛCDM(k) = 1 + A × (k/k_c)^n / (1 + (k/k_c)^m)")
    print("Now showing both A > 0 (enhancement) and A < 0 (suppression) scenarios")
    print()
    
    # Create balanced plots
    plot_balanced_scenarios()
    plot_amplitude_sweep()
    plot_physical_scenarios()
    
    print("\nAll balanced example plots created successfully!")
    print("Key insights:")
    print("- A > 0: Enhancement relative to ΛCDM")
    print("- A < 0: Suppression relative to ΛCDM") 
    print("- A = 0: No deviation (pure ΛCDM)")
    print("- Different DM physics can show both enhancement and suppression")

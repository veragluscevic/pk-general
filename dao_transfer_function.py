#!/usr/bin/env python3
"""
DAO (Dark Acoustic Oscillations) Transfer Function Model

$$T_{\mathrm{IDM}}(k) \equiv \sqrt{\frac{P_{\mathrm{IDM}}(k)}{P_{\Lambda\mathrm{CDM}}(k)}} = T_{\mathrm{env}}(k) \times T_{\mathrm{DAO}}(k)$$

Where:
$$T_{\mathrm{env}}(k) = \left[1 + (\alpha k)^{\beta}\right]^{\gamma}$$
$$T_{\mathrm{DAO}}(k) = 1 + A \sin(k r_s + \phi) \exp\left[-\left(\frac{k}{k_D}\right)^m\right]$$

This is a more physically motivated model for IDM transfer functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os

def dao_transfer_function(k, alpha, beta, gamma, rs, kD, A, phi, m):
    """
    DAO transfer function model.
    
    Parameters:
    -----------
    k : array_like
        Wavenumber in h/Mpc
    alpha : float
        Envelope scale parameter
    beta : float
        Envelope power parameter
    gamma : float
        Envelope power parameter
    rs : float
        Sound horizon scale
    kD : float
        Damping scale
    A : float
        DAO amplitude
    phi : float
        Phase offset
    m : float
        Damping power
        
    Returns:
    --------
    T_IDM : array_like
        Transfer function T_IDM(k)
    """
    
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Envelope function: T_env(k) = [1 + (αk)^β]^γ
    T_env = (1 + (alpha * k_safe)**beta)**gamma
    
    # DAO function: T_DAO(k) = 1 + A sin(k r_s + φ) exp[-(k/k_D)^m]
    T_DAO = 1 + A * np.sin(k_safe * rs + phi) * np.exp(-(k_safe / kD)**m)
    
    T_IDM = T_env * T_DAO
    
    return T_IDM

def envelope_function(k, alpha, beta, gamma):
    """Envelope function only."""
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    return (1 + (alpha * k_safe)**beta)**gamma

def dao_function(k, rs, kD, A, phi, m):
    """DAO function only."""
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    return 1 + A * np.sin(k_safe * rs + phi) * np.exp(-(k_safe / kD)**m)

def plot_dao_model_examples():
    """Plot examples of the DAO model components."""
    
    k = np.logspace(-2, 1.5, 300)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DAO Transfer Function Model\nT_IDM(k) = T_env(k) × T_DAO(k)', fontsize=14, y=0.98)
    
    # Plot 1: Envelope function examples
    ax1 = axes[0, 0]
    envelope_scenarios = [
        {'params': (0.1, 1.0, 0.5), 'label': 'Mild suppression', 'color': 'blue'},
        {'params': (0.2, 1.5, -0.3), 'label': 'Moderate suppression', 'color': 'red'},
        {'params': (0.3, 2.0, -0.5), 'label': 'Strong suppression', 'color': 'green'},
        {'params': (0.15, 1.2, 0.2), 'label': 'Mild enhancement', 'color': 'orange'},
    ]
    
    for scenario in envelope_scenarios:
        alpha, beta, gamma = scenario['params']
        T_env = envelope_function(k, alpha, beta, gamma)
        ax1.semilogx(k, T_env, label=scenario['label'], color=scenario['color'], linewidth=2)
    
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No effect')
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('T_env(k)')
    ax1.set_title('Envelope Function: [1 + (αk)^β]^γ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: DAO function examples
    ax2 = axes[0, 1]
    dao_scenarios = [
        {'params': (1.0, 2.0, 0.1, 0.0, 2.0), 'label': 'Mild oscillations', 'color': 'blue'},
        {'params': (0.5, 1.5, 0.2, 1.0, 1.5), 'label': 'Moderate oscillations', 'color': 'red'},
        {'params': (2.0, 3.0, 0.3, 0.5, 3.0), 'label': 'Strong oscillations', 'color': 'green'},
    ]
    
    for scenario in dao_scenarios:
        rs, kD, A, phi, m = scenario['params']
        T_DAO = dao_function(k, rs, kD, A, phi, m)
        ax2.semilogx(k, T_DAO, label=scenario['label'], color=scenario['color'], linewidth=2)
    
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No oscillations')
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('T_DAO(k)')
    ax2.set_title('DAO Function: 1 + A sin(k r_s + φ) exp[-(k/k_D)^m]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined function examples
    ax3 = axes[1, 0]
    combined_scenarios = [
        {'params': (0.2, 1.5, -0.3, 1.0, 2.0, 0.15, 0.0, 2.0), 'label': 'Suppression + oscillations', 'color': 'red'},
        {'params': (0.15, 1.2, 0.2, 0.8, 1.5, 0.1, 1.0, 1.5), 'label': 'Enhancement + oscillations', 'color': 'blue'},
        {'params': (0.25, 2.0, -0.4, 1.5, 3.0, 0.2, 0.5, 2.5), 'label': 'Strong effects', 'color': 'green'},
    ]
    
    for scenario in combined_scenarios:
        alpha, beta, gamma, rs, kD, A, phi, m = scenario['params']
        T_IDM = dao_transfer_function(k, alpha, beta, gamma, rs, kD, A, phi, m)
        ax3.semilogx(k, T_IDM, label=scenario['label'], color=scenario['color'], linewidth=2)
    
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('T_IDM(k)')
    ax3.set_title('Combined DAO Transfer Function')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter effects
    ax4 = axes[1, 1]
    
    # Show effect of sound horizon rs
    rs_values = [0.5, 1.0, 1.5, 2.0]
    alpha, beta, gamma = 0.2, 1.5, -0.3
    kD, A, phi, m = 2.0, 0.15, 0.0, 2.0
    
    for rs in rs_values:
        T_IDM = dao_transfer_function(k, alpha, beta, gamma, rs, kD, A, phi, m)
        ax4.semilogx(k, T_IDM, label=f'r_s = {rs}', linewidth=2)
    
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='ΛCDM')
    ax4.set_xlabel('k [h/Mpc]')
    ax4.set_ylabel('T_IDM(k)')
    ax4.set_title('Effect of Sound Horizon r_s')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dao_model_examples.png', dpi=300, bbox_inches='tight')
    print("DAO model examples saved as dao_model_examples.png")
    plt.show()

def fit_dao_to_real_data():
    """Fit the DAO model to real transfer function data."""
    
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
    
    # Load LCDM data
    print("Loading LCDM data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Load IDM data
    tk_files = glob.glob('output/*tk.dat')
    tk_files = [f for f in tk_files if 'lcdm' not in f.lower()]
    sample_file = tk_files[10]  # Same file as before
    print(f"Fitting file: {os.path.basename(sample_file)}")
    
    k_idm, d_tot_idm = load_transfer_function_data(sample_file)
    
    # Normalize by LCDM
    d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
    T_normalized = d_tot_idm / d_tot_lcdm_interp
    
    # Filter for k > 0.01
    mask = k_idm > 0.01
    k_filtered = k_idm[mask]
    T_filtered = T_normalized[mask]
    
    print(f"Data points: {len(k_filtered)}")
    print(f"T range: {T_filtered.min():.4f} to {T_filtered.max():.4f}")
    
    # Try multiple initial guesses for DAO model
    initial_guesses = [
        # Guess 1: Based on typical IDM physics
        [0.2, 1.5, -0.3, 1.0, 2.0, 0.15, 0.0, 2.0],
        # Guess 2: Conservative
        [0.1, 1.0, -0.2, 0.8, 1.5, 0.1, 0.0, 1.5],
        # Guess 3: More oscillatory
        [0.15, 1.2, -0.25, 1.5, 3.0, 0.2, 1.0, 2.5],
        # Guess 4: Different scales
        [0.3, 2.0, -0.4, 0.5, 1.0, 0.12, 0.5, 2.0],
        # Guess 5: High frequency oscillations
        [0.18, 1.3, -0.35, 2.0, 4.0, 0.18, 0.0, 3.0],
    ]
    
    # Bounds for DAO model
    bounds = (
        [0.01, 0.1, -1.0, 0.1, 0.1, 0, -np.pi, 0.1],  # Lower bounds
        [1.0, 5.0, 1.0, 10.0, 10.0, 1, np.pi, 5.0]    # Upper bounds
    )
    
    best_fit = None
    best_r2 = -np.inf
    best_params = None
    
    for i, initial_guess in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}: {initial_guess}")
        
        try:
            popt, pcov = curve_fit(dao_transfer_function, k_filtered, T_filtered, 
                                  p0=initial_guess, bounds=bounds, maxfev=15000)
            
            # Calculate R²
            T_pred = dao_transfer_function(k_filtered, *popt)
            ss_res = np.sum((T_filtered - T_pred)**2)
            ss_tot = np.sum((T_filtered - np.mean(T_filtered))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"  R² = {r2:.4f}")
            print(f"  Parameters: α={popt[0]:.3f}, β={popt[1]:.3f}, γ={popt[2]:.3f}")
            print(f"             r_s={popt[3]:.3f}, k_D={popt[4]:.3f}, A={popt[5]:.3f}, φ={popt[6]:.3f}, m={popt[7]:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = popt
                best_fit = (popt, pcov, r2)
                
        except Exception as e:
            print(f"  Fit failed: {e}")
    
    if best_fit is None:
        print("All fits failed!")
        return
    
    print(f"\nBest DAO fit R² = {best_r2:.4f}")
    
    # Plot the best fit
    plt.figure(figsize=(14, 8))
    
    # Plot data
    plt.semilogx(k_filtered, T_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    
    # Generate smooth curve for plotting
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    T_smooth = dao_transfer_function(k_smooth, *best_params)
    plt.semilogx(k_smooth, T_smooth, 'r-', linewidth=2.5, 
                label=f'DAO Fit (R² = {best_r2:.4f})')
    
    # Add reference line
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('T_IDM(k) / T_ΛCDM(k)', fontsize=12)
    plt.title(f'DAO Model Fit: {os.path.basename(sample_file)}', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'DAO Parameters:\nα={best_params[0]:.3f}, β={best_params[1]:.3f}, γ={best_params[2]:.3f}\nr_s={best_params[3]:.3f}, k_D={best_params[4]:.3f}, A={best_params[5]:.3f}\nφ={best_params[6]:.3f}, m={best_params[7]:.3f}'
    
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('dao_fit_to_real_data.png', dpi=300, bbox_inches='tight')
    print("DAO fit to real data saved as dao_fit_to_real_data.png")
    plt.show()
    
    return best_params, best_r2

if __name__ == "__main__":
    print("DAO Transfer Function Model")
    print("=" * 50)
    
    # Plot model examples
    plot_dao_model_examples()
    
    # Fit to real data
    print("\nFitting DAO model to real data...")
    result = fit_dao_to_real_data()
    
    if result is not None:
        params, r2 = result
        print(f"\nDAO fitting completed successfully!")
        print(f"Best R² = {r2:.4f}")
        print("This physically motivated model should work much better!")

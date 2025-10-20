#!/usr/bin/env python3
"""
Refined constrained power spectrum ratio model with oscillation onset closer to suppression.

$$R(k) \equiv \frac{P_{\mathrm{IDM}}(k)}{P_{\Lambda\mathrm{CDM}}(k)}$$
$$R_{\mathrm{env}}(k) = \left[1 + (\alpha k)^{\beta}\right]^{2\gamma}$$

$$\ln R(k) = \ln R_{\mathrm{env}}(k) + S(k) A(k) \sin\Phi(k)$$

Where:
$$S(k) = \frac{1}{2}\left[1+\tanh\left(\frac{\ln(k/k_{\star})}{\Delta}\right)\right]$$
$$A(k) = A_0 \exp\left[-\left(\frac{k}{k_D}\right)^m\right]$$
$$\Phi(k) = \phi_0 + \int_{k_{\star}}^{k}\omega(u)du$$
$$\omega(u) = r_s\left[1+\epsilon\ln\left(\frac{u}{k_{\star}}\right)\right]$$

With constraint: $k_{\star} = k_{\mathrm{star\_factor}} \times \frac{1}{\alpha}$
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
import glob
import os

def refined_constrained_model(k, alpha, beta, gamma, k_star_factor, delta, A_0, k_D, m, phi_0, r_s, epsilon):
    """
    Refined constrained power spectrum ratio model.
    
    k_star_factor: multiplicative factor for suppression scale (k_star = k_star_factor * k_suppression)
    This allows oscillations to start closer to the suppression onset.
    """
    k = np.asarray(k)
    k_safe = np.maximum(k, 1e-10)
    
    # Envelope function: R_env(k) = [1 + (αk)^β]^(2γ)
    R_env = (1 + (alpha * k_safe)**beta)**(2*gamma)
    
    # Calculate suppression scale more carefully
    # For R_env = [1 + (αk)^β]^(2γ), we want the scale where R_env starts deviating from 1
    # Let's use a more sophisticated approach: find where R_env drops to some threshold
    # For now, use k_suppression = 0.5/α as a better estimate
    k_suppression = 0.5 / alpha
    
    # Oscillation onset: k_star = k_star_factor * k_suppression
    k_star = k_star_factor * k_suppression
    
    # Transition function: S(k) = 0.5[1 + tanh(ln(k/k_star)/Δ)]
    S = 0.5 * (1 + np.tanh(np.log(k_safe / k_star) / delta))
    
    # Amplitude function: A(k) = A_0 * exp[-(k/k_D)^m]
    A = A_0 * np.exp(-(k_safe / k_D)**m)
    
    # Phase function: Φ(k) = φ_0 + ∫[k_star to k] ω(u) du
    k_integration = np.linspace(k_star, k_safe.max(), 1000)
    omega = r_s * (1 + epsilon * np.log(k_integration / k_star))
    phase_integration = cumulative_trapezoid(omega, k_integration, initial=0)
    Phi = phi_0 + np.interp(k_safe, k_integration, phase_integration)
    
    # Power spectrum ratio in log space
    ln_R = np.log(R_env) + S * A * np.sin(Phi)
    
    # Convert back to linear space
    R = np.exp(ln_R)
    
    return R

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

def test_refined_constrained_model():
    """Test the refined constrained model."""
    
    # Load data
    print("Loading data...")
    k_lcdm, d_tot_lcdm = load_transfer_function_data('output/pk-lcdm_tk.dat')
    
    # Use the oscillatory file
    sample_file = 'output/pk_m4.6e-04_sig1.0e-25_np2_tk.dat'
    print(f"Testing refined constrained model on: {os.path.basename(sample_file)}")
    
    k_idm, d_tot_idm = load_transfer_function_data(sample_file)
    
    # Calculate power spectrum ratio R(k) = T_IDM²(k) / T_ΛCDM²(k)
    d_tot_lcdm_interp = np.interp(k_idm, k_lcdm, d_tot_lcdm)
    T_ratio = d_tot_idm / d_tot_lcdm_interp
    R = T_ratio**2  # Power spectrum ratio
    
    # Filter for k > 0.01
    mask = k_idm > 0.01
    k_filtered = k_idm[mask]
    R_filtered = R[mask]
    
    print(f"Data points: {len(k_filtered)}")
    print(f"k range: {k_filtered.min():.3f} to {k_filtered.max():.3f}")
    print(f"R range: {R_filtered.min():.4f} to {R_filtered.max():.4f}")
    
    # Initial parameter guesses for the refined model
    # k_star_factor should be close to 1.0 (oscillations start near suppression)
    initial_guesses = [
        # Guess 1: Oscillations start very close to suppression
        [0.1, 1.5, -0.5, 1.2, 0.2, 0.5, 1.0, 2.0, 0.0, 1.0, 0.1],
        # Guess 2: Slightly later onset
        [0.2, 2.0, -0.3, 1.5, 0.1, 0.3, 0.8, 1.5, 0.5, 0.8, 0.05],
        # Guess 3: Conservative
        [0.15, 1.0, -0.4, 1.3, 0.15, 0.4, 1.2, 2.5, 0.0, 1.2, 0.08],
        # Guess 4: Very close onset
        [0.1, 1.2, -0.3, 1.1, 0.1, 0.2, 0.5, 1.0, 0.0, 0.5, 0.0],
        # Guess 5: Try different envelope parameters
        [0.05, 3.0, -0.6, 1.4, 0.2, 0.6, 1.5, 2.0, 0.0, 1.5, 0.1],
    ]
    
    # Parameter bounds for refined model
    # k_star_factor: should be close to 1.0 (oscillations start near suppression)
    bounds = (
        [0.001, 0.1, -2.0, 0.8, 0.01, 0.001, 0.01, 0.1, -np.pi, 0.01, -1.0],  # Lower bounds
        [1.0, 5.0, 0.0, 3.0, 1.0, 2.0, 10.0, 10.0, np.pi, 10.0, 1.0]       # Upper bounds
    )
    
    print("\nRefined constrained model parameters:")
    print("α, β, γ: envelope function parameters")
    print("k_star_factor: multiplicative factor for suppression scale")
    print("  k_star = k_star_factor * k_suppression (should be close to 1.0)")
    print("δ: transition width")
    print("A_0: oscillation amplitude")
    print("k_D, m: damping parameters")
    print("φ_0: initial phase")
    print("r_s: sound horizon")
    print("ε: frequency evolution")
    
    best_fit = None
    best_r2 = -np.inf
    best_params = None
    
    for i, initial_guess in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}: {initial_guess}")
        
        try:
            popt, pcov = curve_fit(refined_constrained_model, k_filtered, R_filtered, 
                                  p0=initial_guess, bounds=bounds, maxfev=50000)
            
            # Calculate R²
            R_pred = refined_constrained_model(k_filtered, *popt)
            ss_res = np.sum((R_filtered - R_pred)**2)
            ss_tot = np.sum((R_filtered - np.mean(R_filtered))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate actual scales
            alpha = popt[0]
            k_star_factor = popt[3]
            k_suppression = 0.5 / alpha
            k_star = k_star_factor * k_suppression
            
            print(f"  R² = {r2:.4f}")
            print(f"  Parameters: α={popt[0]:.3f}, β={popt[1]:.3f}, γ={popt[2]:.3f}")
            print(f"             k_star_factor={popt[3]:.3f}, δ={popt[4]:.3f}, A_0={popt[5]:.3f}")
            print(f"             k_D={popt[6]:.3f}, m={popt[7]:.3f}, φ_0={popt[8]:.3f}")
            print(f"             r_s={popt[9]:.3f}, ε={popt[10]:.3f}")
            print(f"             k_suppression={k_suppression:.3f}, k_star={k_star:.3f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = popt
                best_fit = (popt, pcov, r2)
                
        except Exception as e:
            print(f"  Fit failed: {e}")
    
    if best_fit is None:
        print("All fits failed!")
        return
    
    print(f"\nBest refined constrained model fit R² = {best_r2:.4f}")
    
    # Plot the results
    plt.figure(figsize=(16, 12))
    
    # Main plot - Power spectrum ratio
    plt.subplot(2, 3, 1)
    plt.semilogx(k_filtered, R_filtered, 'o', alpha=0.6, markersize=4, 
                label='Data', color='black')
    
    k_smooth = np.logspace(np.log10(k_filtered.min()), np.log10(k_filtered.max()), 300)
    R_smooth = refined_constrained_model(k_smooth, *best_params)
    plt.semilogx(k_smooth, R_smooth, 'r-', linewidth=2.5, 
                label=f'Refined Model (R² = {best_r2:.4f})')
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1, label='ΛCDM')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('R(k) = P_IDM(k) / P_ΛCDM(k)')
    plt.title(f'Refined Constrained Model: {os.path.basename(sample_file)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Components breakdown
    alpha, beta, gamma, k_star_factor, delta, A_0, k_D, m, phi_0, r_s, epsilon = best_params
    
    # Calculate actual scales
    k_suppression = 0.5 / alpha
    k_star = k_star_factor * k_suppression
    
    # Calculate components
    R_env = (1 + (alpha * k_smooth)**beta)**(2*gamma)
    S = 0.5 * (1 + np.tanh(np.log(k_smooth / k_star) / delta))
    A = A_0 * np.exp(-(k_smooth / k_D)**m)
    
    # Phase calculation
    k_integration = np.linspace(k_star, k_smooth.max(), 1000)
    omega = r_s * (1 + epsilon * np.log(k_integration / k_star))
    phase_integration = cumulative_trapezoid(omega, k_integration, initial=0)
    Phi = phi_0 + np.interp(k_smooth, k_integration, phase_integration)
    
    # Plot envelope
    plt.subplot(2, 3, 2)
    plt.semilogx(k_smooth, R_env, 'b-', linewidth=2, label='R_env(k)')
    plt.axvline(x=k_suppression, color='blue', linestyle=':', alpha=0.7, label=f'k_suppression={k_suppression:.3f}')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No effect')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('R_env(k)')
    plt.title(f'Envelope: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot transition function
    plt.subplot(2, 3, 3)
    plt.semilogx(k_smooth, S, 'g-', linewidth=2, label='S(k)')
    plt.axvline(x=k_suppression, color='blue', linestyle=':', alpha=0.7, label=f'k_suppression={k_suppression:.3f}')
    plt.axvline(x=k_star, color='red', linestyle=':', alpha=0.7, label=f'k_star={k_star:.3f}')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('S(k)')
    plt.title(f'Transition: k_star_factor={k_star_factor:.3f}, δ={delta:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot amplitude
    plt.subplot(2, 3, 4)
    plt.semilogx(k_smooth, A, 'purple', linewidth=2, label='A(k)')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('A(k)')
    plt.title(f'Amplitude: A_0={A_0:.3f}, k_D={k_D:.3f}, m={m:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot phase
    plt.subplot(2, 3, 5)
    plt.semilogx(k_smooth, Phi, 'orange', linewidth=2, label='Φ(k)')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Φ(k)')
    plt.title(f'Phase: φ_0={phi_0:.3f}, r_s={r_s:.3f}, ε={epsilon:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 3, 6)
    R_pred = refined_constrained_model(k_filtered, *best_params)
    residuals = R_filtered - R_pred
    plt.semilogx(k_filtered, residuals, 'o', alpha=0.6, markersize=3, color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('Residuals')
    plt.title(f'Residuals (RMS = {np.sqrt(np.mean(residuals**2)):.4f})')
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    param_text = f'Refined Model:\nα={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}\nk_star_factor={k_star_factor:.3f}, δ={delta:.3f}, A_0={A_0:.3f}\nk_D={k_D:.3f}, m={m:.3f}, φ_0={phi_0:.3f}\nr_s={r_s:.3f}, ε={epsilon:.3f}\n\nk_suppression={k_suppression:.3f}\nk_star={k_star:.3f}'
    
    plt.figtext(0.02, 0.02, param_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('refined_constrained_model_fit.png', dpi=300, bbox_inches='tight')
    print("Refined constrained model fit saved as refined_constrained_model_fit.png")
    plt.show()
    
    return best_params, best_r2, k_suppression, k_star

if __name__ == "__main__":
    print("Testing the refined constrained power spectrum ratio model...")
    print("=" * 70)
    
    result = test_refined_constrained_model()
    
    if result is not None:
        params, r2, k_suppression, k_star = result
        print(f"\nRefined constrained model testing completed!")
        print(f"Best R² = {r2:.4f}")
        print(f"Suppression onset: k_suppression = {k_suppression:.3f}")
        print(f"Oscillation onset: k_star = {k_star:.3f}")
        print("Oscillations now start closer to suppression onset!")

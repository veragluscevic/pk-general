import numpy as np
import os

main_root = '/Users/veragluscevic/research/repositories/pk-general'

# Define parameter ranges
m_dmeff_values = np.logspace(-5, 0, 10)
sigma_dmeff_values = np.logspace(-30, -25, 10)
npow_dmeff_values = np.array([0, 2, 4])

# Input and output directories
base_ini = main_root+"/base.ini"
output_dir = os.path.join(main_root,"inis")
os.makedirs(output_dir, exist_ok=True)

# Load the base ini file
with open(base_ini, "r") as f:
    base_content = f.read()

# Loop over parameter combinations
for npow in npow_dmeff_values:
    for m in m_dmeff_values:
        for sigma in sigma_dmeff_values:
            new_content = base_content
            new_content = new_content.replace("m_dmeff = 0.01", f"m_dmeff = {m:.6e}")
            new_content = new_content.replace("sigma_dmeff = 1e-28", f"sigma_dmeff = {sigma:.6e}")
            new_content = new_content.replace("npow_dmeff = 0", f"npow_dmeff = {npow}")

            # Update root name to reflect parameters
            root_name = f"pk_m{m:.1e}_sig{sigma:.1e}_np{npow}"
            new_content = new_content.replace(
                f"root = {main_root}/output/pk-idm-n0-0.01GeV-1e-28",
                f"root = {main_root}/output/{root_name}"
            )

            # Write to file
            out_name = f"pk-m{m:.1e}_sigma{sigma:.1e}_np{npow}.ini"
            out_path = os.path.join(output_dir, out_name)
            with open(out_path, "w") as f:
                f.write(new_content)

print(f"Created ini files in {output_dir}/")

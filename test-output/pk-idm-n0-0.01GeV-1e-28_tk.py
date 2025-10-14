import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/Users/veragluscevic/research/repositories/pk-general/output/pk-idm-n0-0.01GeV-1e-28_tk.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['pk-idm-n0-0']

fig, ax = plt.subplots()

index, curve = 0, data[0]
y_axis = ['d_g', 'd_b', 'd_cdm', 'd_dmeff', 'd_ur', 'd_tot', 'phi', 'psi']
tex_names = ['d_g', 'd_b', 'd_cdm', 'd_dmeff', 'd_ur', 'd_tot', 'phi', 'psi']
x_axis = 'k (h/Mpc)'
ylim = []
xlim = []
ax.loglog(curve[:, 0], abs(curve[:, 1]))
ax.loglog(curve[:, 0], abs(curve[:, 2]))
ax.loglog(curve[:, 0], abs(curve[:, 3]))
ax.loglog(curve[:, 0], abs(curve[:, 4]))
ax.loglog(curve[:, 0], abs(curve[:, 5]))
ax.loglog(curve[:, 0], abs(curve[:, 6]))
ax.loglog(curve[:, 0], abs(curve[:, 7]))
ax.loglog(curve[:, 0], abs(curve[:, 8]))

ax.legend([root+': '+elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('k (h/Mpc)', fontsize=16)
plt.show()
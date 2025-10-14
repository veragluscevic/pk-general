import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/Users/veragluscevic/research/repositories/pk-general/output/pk-idm-n0-0.01GeV-1e-28_pk.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['pk-idm-n0-0']

fig, ax = plt.subplots()

index, curve = 0, data[0]
y_axis = ['P(Mpc/h)^3']
tex_names = ['P (Mpc/h)^3']
x_axis = 'k (h/Mpc)'
ylim = []
xlim = []
ax.loglog(curve[:, 0], abs(curve[:, 1]))

ax.legend([root+': '+elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('k (h/Mpc)', fontsize=16)
plt.show()
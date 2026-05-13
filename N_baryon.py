import numpy as np
import matplotlib.pyplot as plt

z_0 = np.linspace(15., 60., 901)
n_b0 = np.load('./Avg_n_Baryon_z.npy')
print(n_b0, n_b0.shape)
z = np.linspace(60., 200., 141)
Dz = np.load('./Dz_Dark_Ages.npy')
n_0 = 2.471e-7

n_b = np.zeros(len(z))
for iz in range(0, len(z)):
    z_pres = round(z[iz],1)
    n_b[iz] = n_0 * (1.+z_pres)**3.
print(n_b)
np.save('./Avg_n_Baryon_Dark_Ages.npy', n_b)

plt.figure()
plt.semilogy(z_0, n_b0, label='Fiducial')
plt.semilogy(z, n_b, label='Dark Ages')
plt.xlim(200., 15.)
plt.xlabel('Redshift')
plt.ylabel('n_baryon')
plt.legend()
plt.savefig('./Baryon.pdf')


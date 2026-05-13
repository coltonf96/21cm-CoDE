import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

z = np.linspace(15., 70., 1101)                 #Extended redshift array
H_0, Omega_m, c = 67., 0.32, 29979245800.       #Various constants
H_0_s, h, Omega_L = H_0/3.0857e19, H_0/100., 1.0 - Omega_m
z_max_all = np.load('./Ly_alpha_z_max.npy')     #Max redshift from which photons reach current redshifts (by hydrogen level n)
ns = np.arange(2,24)                            #Hydrogen energy levels
print(ns)

d0 = 3.0                                        #Distance from center of one cell to nearest neighbors sharing one full side (6 cells)
d1 = 2. * np.sqrt(1.5**2. + 1.5**2.)            #Distance from center of cell to neighbors sharing one full edge (12 cells)
d2 = 2. * np.sqrt(1.5**2. + 1.5**2. + 1.5**2.)  # "" to neighbors that only touch at the corners (8 cells)
test0 = np.array((d0, d1, d2))                  #Base distances to form d(z) array

plt.figure()
colors = plt.cm.jet(np.linspace(0,1,len(z)))
all_dists_z = []                      #Initialize grand array of z_max(z_pres, n)
for iz in range(0, 901):              #Loop through time
  z_pres = round(z[iz],2)             #Current redshift
#  print(z_pres)
  z_max_n = z_max_all[iz]             #Max z's from which photons of each H level can reach present z
  z_highest = z_max_n[-1]             #Furthest redshift for n = 2 transition (max overall for this z)
  all_dists = np.zeros(len(z_max_n))  #Initialize distance array to each z_max(n)
  for N in range(0, len(z_max_n)):                #Now loop through H energy levels from 2-23
    z_max = z_max_n[N]                                                      #Current E-level maximum redshift
    z_higher = np.linspace(z_pres, z_max, 1000)                             #Array of redshifts out to z_max(n)
    H = 2.333e-18 * (h/0.72) * np.sqrt(Omega_m*(1.+z_higher)**3. + Omega_L) #H(z) used in getting distances
    d = scipy.integrate.cumtrapz(c/H, z_higher) / (3.0857e24)               #Distances (r_cMpc from Ahn09)
    all_dists[N] = d[-1]                                                    #Record d(z_max) for this n
  all_dists_z.append(all_dists)       #Append the grand list with z_max distances for all n at z_pres
  if iz % 100 == 0:
    print(z_pres)
    print(np.column_stack((z_max_n, all_dists)))
    plt.semilogy(ns[:-1], all_dists, color=colors[iz], label=str(z_pres))
np.save('./Ly_alpha_Smoothing_Scales.npy', np.array(all_dists_z), allow_pickle=True)
plt.axhline(3., color='black', ls='--', label='Cell (3 Mpc)') 
plt.axhline(192., color='black', ls=':', label='Sim Box (192 Mpc)')
plt.xlabel('Hydrogen energy level n')
plt.ylabel('Distance to z_max(n) (cMpc)')
plt.legend(fontsize=9)

plt.savefig('./Ly-a_Max_Distances.pdf')

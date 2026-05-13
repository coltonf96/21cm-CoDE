import numpy as np
import scipy.integrate

#z = np.linspace(15., 60., 901)              #Redshift array
z = np.linspace(60., 200., 141)
special = '_Dark_Ages'
H_0, Omega_m, c = 67., 0.32, 29979245800.   #Hubble param (km/s/Mpc), cosmo matter density param, speed of light (cm/s)
h, Omega_L = H_0/100., 1.0 - Omega_m        #Unitless Hubble param, cosmo DE density parameter
all_d = []                                  #Initialize all distances to previous z's for all z_present

for iz in range(0, len(z)):                     #Loop through time
    z_pres, z_higher = round(z[iz],2), z[iz:]                               #Current redshift, and array of higher z values
    H = 2.333e-18 * (h/0.72) * np.sqrt(Omega_m*(1.+z_higher)**3. + Omega_L) #H(z), converted to 1/s
    d = scipy.integrate.cumtrapz(c/H, z_higher) / (3.0857e24)               #Distances, converted from cm to r_cMpc (from Ahn09)
    d = np.concatenate(([0.], d))                                           #Append distance array to include distance to z_pres = 0 
    all_d.append(d)                             #Record distances(z)
    print(z_pres, d[0:4], len(d))
np.save('./All_z_Distances' + special + '.npy', np.array(all_d, dtype=object), allow_pickle=True)    #Save results

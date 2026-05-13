import numpy as np
import matplotlib.pyplot as plt

N = 1000
masses = np.logspace(0,16.1,N)
h, omm = 0.67, 0.32
boxsizes = [1., 3., 5., 10., 50., 192.] #Box side lengths in h^-1 Mpc
rhoc = 1.36e11*(h/0.7)**2.              #Critical density today in Solar Mass per Mpc^3 
rhom = rhoc*omm
trans = np.loadtxt('./trans_planck.dat')#Transfer function - get power spectrum P(k)  
kbig = trans[:,0]*h

for iv in range(0, len(boxsizes)):      #Loop through each box size
    boxsize = boxsizes[iv]              #Current box side length
    print(boxsize)
    Rs, sigmas = np.zeros([N]), np.zeros([N])   #Initialize arrays of R-values & sigmas
    kmin = np.pi/(boxsize*np.sqrt(3.)/h)        #Minimum k-mode that can fit in the volume (max distance scale)
    Pbig = (0.83/2.6721e-4)**2. * trans[:,0]**0.96 * trans[:,1]**2.
    Pbig = np.where(kbig < kmin, 0, Pbig)
    for i in range(N):                          #Loop through halo masses
        M = masses[i]
        Rs[i] = (3./4./np.pi*M/rhom)**(1./3.)   #Calculate equivalent halo radius to get f_int
        fint = 1./(2.*np.pi**2.) * kbig**2 * Pbig * (3*(np.sin(kbig*Rs[i])-(kbig*Rs[i])*np.cos(kbig*Rs[i]))/(kbig*Rs[i])**3.)**2.
        sigmas[i] = np.trapz(kbig*fint, np.log(kbig))**0.5      #Calculate & record sigma for this mass bin
    plt.loglog(masses, sigmas)
    np.save('./Sigmas_Box_{:.1f}.npy'.format(boxsize), sigmas)
plt.show()

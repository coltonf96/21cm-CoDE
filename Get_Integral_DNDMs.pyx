import numpy as np
cimport numpy as np
cimport cython

def get_sigma(double Mass, np.ndarray[np.float64_t,ndim=1] masses, np.ndarray[np.float64_t,ndim=1] sigmas):
    return np.interp(np.log(Mass), np.log(masses), sigmas)

def Get_dndM(np.ndarray[np.float64_t,ndim=1] mass_bins, double z, np.ndarray[np.float64_t,ndim=1] sigmas, double delta_x, np.ndarray[np.float64_t,ndim=1] params, np.ndarray[np.float64_t,ndim=1] masses):
    cdef int i
    cdef double A_p, a_p, q_p, dm, rho_c, rho_0, delta_c, S, S_dm, nu, f_st, f_ps_denom, M_box, S_R, f_ps_numer, dS_dM
    cdef np.ndarray[np.float64_t,ndim=1] DNDM

    A_p, a_p, q_p, dm, rho_c, rho_0, delta_c = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
    DNDM = np.zeros(len(mass_bins))         #Initialize dn/dM
    for i in range(0, len(mass_bins)):      #Loop through halo masses to get dn/dM(M)
        S, S_dm = (get_sigma(mass_bins[i],masses,sigmas))**2., (get_sigma(mass_bins[i]+dm,masses,sigmas))**2.   #Variance of M_halo & M_halo+dm
        nu = delta_c/(np.sqrt(S))
        f_st = A_p*(nu/S) * np.sqrt(a_p/(2.*np.pi)) * (1. + 1./((a_p*nu**2.)**q_p)) * np.exp(-(a_p*nu**2.)/2.)  #Sheth-Tormen
        f_ps_denom, M_box = delta_c/(S**1.5) * np.exp(-(delta_c**2.)/(2.*S)), 27.*rho_0                         #Press-Schechter and mass of (3 Mpc)^3 box at this overdensity
        S_R = (get_sigma(M_box,masses,sigmas))**2.                                                      #And variance at the scale of that box
        if S < S_R or f_ps_denom < 1e-300:          #Stop whenever we drop below resolution scale
          break
        f_ps_numer = (delta_c-delta_x)/((S-S_R)**1.5) * np.exp(-0.5*((delta_c-delta_x)**2.)/(S-S_R))    #Biased Press-Schechter
        dS_dM = np.abs((S_dm - S)/dm)
        DNDM[i] = (rho_0/mass_bins[i])*dS_dM*f_st*(f_ps_numer/f_ps_denom)   #Calculate & record dn/dM(z)
        if np.log10(DNDM[i]) < -300.:                                       #Make any negligible values zero for simplicity
          DNDM[i+1:] = 0.0
          break
    return DNDM

def Run(np.ndarray[np.float64_t,ndim=1] overdensities, double l_cell, int iz_0, int iz_N, str special=''):
    cdef int iz, delta
    cdef double V_com, A_p, a_p, q_p, dm, G, H_0, Omega_m, Omega_l, rho_c, rho_0, z_pres, a, why, Dz, delta_c, delta_x
    cdef np.ndarray[np.float64_t,ndim=1] z, sigmas, M_crit_0, masses, ap, params, masses_int
    cdef np.ndarray[np.float64_t,ndim=2] all_DNDM

    z = np.linspace(15., 60., 901)      #Redshift array
    sigmas = np.load('./Sigmas.npy')    #Halo mass variances (global)
    if l_cell > 1e-8:                   #For finite volumes, load in truncated Sigmas array instead
        sigmas = np.load('./Sigmas_Box_' + str(round(l_cell,1)) + '.npy')
    M_crit_0 = np.load('./J_0/J_0_0.0.npy') #Lowest possible M_crit(z), for vbc=J_LW=0

    A_p, a_p, q_p, dm, G, H_0, Omega_m = 0.322, 0.75, 0.3, 100., 4.494e-33, 6.847e-11, 0.32 #Parameters for getting dn/dM (G in Mpc^3 M_sun^-1 yr^-2, H_0 in yr^-1)
    Omega_l, rho_c, V_com = 1.- Omega_m, (3.*H_0**2.)/(8.*np.pi*G), 27.                     #DE fraction & Critical density, cell volume (Mpc^3)
    rho_0, masses = Omega_m * rho_c, np.logspace(0,16.1,1000)                               #Matter density & integration array

    for iz in range(iz_0, iz_N):            #Loop through redshifts
        z_pres = round(z[iz],2)             #Current redshift
        a = 1./(1.+z_pres)                  #Scale factor at current z 
        ap = np.linspace(0.0, a, 10000)     #Array of scale factors from present (0) to a
        why = np.sqrt(Omega_l*a**3. + Omega_m) / (a**1.5) * np.trapz((ap/(Omega_l*ap**3. + Omega_m))**1.5, ap)
        Dz = why/1.125940274656245/0.877787481277015                  #Normalize growth factor such that D(0) = 1
        delta_c = 1.686/Dz                                            #Current critical overdensity for collapse
        params = np.array([A_p, a_p, q_p, dm, rho_c, rho_0, delta_c]) #Update parameters array with current critical density
        masses_int = np.logspace(np.log10(M_crit_0[iz]), 13., 750)    #Finely spaced array to be integrated over to get dn/dM
        all_DNDM = np.zeros([len(masses_int), len(overdensities)])    #Initialize large array containing all dn/dM arrays for current z
        print(z_pres)
        for delta in range(0, len(overdensities)):      #Now loop through all overdensity bins & calculate dn/dM(z_pres) for each
            delta_x = overdensities[delta]              #Current overdensity bin value
            all_DNDM[:,delta] = Get_dndM(masses_int, z_pres, sigmas, delta_x, params, masses)
        print(all_DNDM[:,-1])
        np.save('./Int_DNDMs/Num_Halos_z_' + str(z_pres) + special + '.npy', all_DNDM)
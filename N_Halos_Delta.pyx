import numpy as np
cimport numpy as np
cimport cython
 
def get_sigma(double Mass, np.ndarray[np.float64_t,ndim=1] masses, np.ndarray[np.float64_t,ndim=1] sigmas):
    return np.interp(np.log(Mass), np.log(masses), sigmas)

def Get_dndM(np.ndarray[np.float64_t,ndim=1] mass_bins, double z, np.ndarray[np.float64_t,ndim=1] sigmas, double delta_x, np.ndarray[np.float64_t,ndim=1] params):
    cdef int i
    cdef double A_p, a_p, q_p, dm, rho_c, rho_0, delta_c, S, S_dm, nu, f_st, f_ps_denom, M_box, S_R, f_ps_numer, dS_dM, Dz, delta_xz
    cdef np.ndarray[np.float64_t,ndim=1] DNDM, masses
    
    A_p, a_p, q_p, dm, rho_c, rho_0, delta_c, Dz = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    DNDM, masses = np.zeros(len(mass_bins)), np.logspace(0,16.1,1000)   #Initialize dn/dM & integral array    
    delta_xz = delta_x/Dz
    for i in range(0, len(mass_bins)):                                  #Loop through halo masses to get dn/dM(M)
        S, S_dm = (get_sigma(mass_bins[i],masses,sigmas))**2., (get_sigma(mass_bins[i]+dm,masses,sigmas))**2.   #Sigma variances for M & M+dm
        nu = delta_c/(np.sqrt(S))
        f_st = A_p*(nu/S) * np.sqrt(a_p/(2.0*np.pi)) * (1. + 1./((a_p*nu**2.)**q_p)) * np.exp(-(a_p*nu**2.)/2.) #Sheth-Tormen
        f_ps_denom = delta_c/(S**1.5) * np.exp(-(delta_c**2.)/(2.*S))                                           #Press-Schechter
        M_box = (3**3.)*rho_0                                           #Mass of (3 Mpc)^3 box at this overdensity
        S_R = (get_sigma(M_box,masses,sigmas))**2.      #And variance at the scale of that box
        if S < S_R:                                     #Break if we fall below resolution scale
          break
        f_ps_numer = (delta_c-delta_xz)/((S-S_R)**1.5) * np.exp(-0.5 * ((delta_c-delta_xz)**2.)/(S-S_R)) #Biased Press-Schechter
        dS_dM = np.abs((S_dm - S)/dm)
        DNDM[i] = (rho_0/mass_bins[i])*dS_dM*f_st*(f_ps_numer/f_ps_denom)
    return DNDM

def Get_N_Halos(double z, np.ndarray[np.float64_t,ndim=1] mass_bins, np.ndarray[np.float64_t,ndim=1] mbin_l, np.ndarray[np.float64_t,ndim=1] mbin_r, double V_com, np.ndarray[np.float64_t,ndim=1] sigmas, double delta_x, np.ndarray[np.float64_t,ndim=1] params):
    cdef int Nintegral, im
    cdef np.ndarray[np.float64_t,ndim=1] N_halos, masses, DNDM_int

    N_halos = np.zeros(len(mass_bins))                          #Initialize Num_Halos per bin
    print('Getting dn/dM for delta(x) = ', delta_x)
    for im in range(0, len(mass_bins)):                         #Now loop through each mass bin
        masses = np.linspace(mbin_l[im], mbin_r[im], 1000)      #Using bounds around this bin to get integrand array
        DNDM_int = Get_dndM(masses, z, sigmas, delta_x, params) #Get actual dn/dM value
        N_halos[im] = V_com*np.trapz(DNDM_int, masses)          #Integrate and multiply by box size to get N_halos
        if N_halos[im] < 1e-300:                                #Stop once halo abundances become negligible
          break
    print(np.column_stack((mass_bins, N_halos)))
    return(N_halos)

def Run(np.ndarray[np.float64_t,ndim=1] overdensities, double z_i, double L_side, str special='', int globally=0):
    cdef int delta
    cdef double V_com, A_p, a_p, q_p, dm, G, H_0, Omega_m, Omega_l, rho_c, rho_0, a, log_0, log_1, log_N, diff
    cdef np.ndarray[np.float64_t,ndim=1] z, mass_bins, Dz_all, sigmas, N_Halos, mbin_center, mbin_l, mbin_r

    z = np.linspace(15., 60., 901)          #Fiducial redshift array
    mass_bins = np.logspace(5.6, 9.5, 40)   #Fiducial halo mass bins
    print(np.log10(mass_bins))
    if globally == 0:                       #Load in sigmas values for finite volume
      sigmas = np.load('./Sigmas_Box_' + str(round(L_side,1)) + '.npy')
      V_com = L_side**3.                    #Corresponding box volume
    else:
      sigmas = np.load('./Sigmas.npy')      #Or load in global sigma values for global HMF
      V_com = 1.                            #Volume = 1 Mpc^3 for global

    Dz_all = np.load('./Dz_15.npy')         #All growth factors D(z)
    Dz = Dz_all[np.argmin(np.abs(z-z_i))]   #Growth factor at z_i
    log_0, log_1, log_N = np.log10(mass_bins[0]), np.log10(mass_bins[1]), np.log10(mass_bins[len(mass_bins)-1])
    diff = (log_1 - log_0)/2.0                                                          #Take logs of 1st, 2nd, and last bins to get bin centers
    mbin_center = np.logspace(log_0-diff, log_N+diff, len(mass_bins)+1)                 #Mass bin centers & edges
    mbin_l, mbin_r = mbin_center[0:len(mbin_center)-1], mbin_center[1:len(mbin_center)] #Isolate left and right bounds

    A_p, a_p, q_p, dm, G, H_0, Omega_m = 0.322, 0.75, 0.3, 100., 4.494e-33, 6.847e-11, 0.32 #Parameters for getting dn/dM
    rho_c = (3.*H_0**2.)/(8.*np.pi*G)       #DE fraction & Critical density
    rho_0 = Omega_m*rho_c                   #Matter density parameter & scale factor at z
    delta_c = 1.686/Dz                      #Current critical overdensity
    params = np.array([A_p, a_p, q_p, dm, rho_c, rho_0, delta_c, Dz])

    for delta in range(0, len(overdensities)):  #Loop through each delta bin & calculate HMFs
        N_Halos = Get_N_Halos(z_i, mass_bins, mbin_l, mbin_r, V_com, sigmas, overdensities[delta], params)
        np.save('./HMFs/Num_Halos_' + str(delta) + special + '.npy', N_Halos)
    print(mass_bins)
    print(N_Halos)

import numpy as np
import os, time, scipy
cimport numpy as np
cimport cython

# This function is for using a spherical top-hat to smooth over the SFR cubes using FFTs --------------------------------------------
def smooth_cube(cube, radius, avg):
    cdef int cube_size
    cdef list dists, corners
    cdef np.ndarray[np.int_t,ndim=1] work_shape
    cdef np.ndarray[np.complex128_t,ndim=3] cube_k, kernel_k, result0
    cdef np.ndarray[np.float64_t,ndim=3] work_cube, dist, kernel, result

    work_cube = cube.copy()                             #Add padding around edges of cube to remove periodic B.C.s
    cube_k = np.fft.fftn(work_cube)                     #Take the FFT of this padded cube
    cube_size = len(work_cube[:,0,0])                   #Save the cube size
    dists = []
    xg, yg, zg = np.meshgrid(np.arange(cube_size), np.arange(cube_size), np.arange(cube_size))              #Get distances from center cube & corner cell IDs
    corners = [[0,0,0],[0,0,cube_size],[0,cube_size,0],[0,cube_size,cube_size],[cube_size,0,0],[cube_size,0,cube_size],[cube_size,cube_size,0],[cube_size,cube_size,cube_size]]
    for i in range(len(corners)):                                                                           #Loop through and get distances from center to each corner cell
        dists.append(np.sqrt((corners[i][0]-xg)**2. + (corners[i][1]-yg)**2. + (corners[i][2]-zg)**2.))     #Append list
    dist = np.amin(dists, axis=0)                                                                           #Now make a cube of distances from each cell to the nearest corner
    work_shape = np.array((work_cube.shape[0], work_cube.shape[1], work_cube.shape[2]))                     #Get lengths of sides to initialize kernel
    kernel = np.zeros(work_shape)                       #Initialize the kernel as an equal size cube of zeros...
    kernel[(dist < radius)] = 1                         #Set any point within radius from the center to be 1
    if avg == 1:                                        #If we're getting the average...
        kernel /= np.sum(kernel)                        #Divide the kernel by the number of points to get an average
    kernel_k = np.fft.fftn(kernel)                      #FFT the kernel cube
    result0 = np.fft.ifftn(cube_k * kernel_k)           #Now that we have the kernel, perform the convolution
    result = np.real(result0)                           #Remove imaginary components
    result[result < 1e-15] = 0.0                        #And any negative values/numerical errors
    del work_cube, cube_k, kernel, kernel_k, result0
    return(result)

# This function is for smoothing the SFR values over the lifetime of the star -------------------------------------------------------
def Smooth_SFRs(int N, int z_iz, np.ndarray[np.float64_t,ndim=1] z, np.ndarray[np.float64_t,ndim=1] t, np.ndarray[np.float64_t,ndim=4] SFR_III_all, np.ndarray[np.float64_t,ndim=4] SFR_II_all, double t_life): 
    cdef int iz, i, t_iz, N_steps
    cdef double t_3
    cdef np.ndarray[np.float64_t,ndim=1] z_long, t_long
    cdef np.ndarray[np.float64_t,ndim=3] SFR_II_z, SFR_III_z, SFR_II_add, SFR_III_add
    cdef np.ndarray[np.float64_t,ndim=4] SFR_II_smooth, SFR_III_smooth

    z_long = np.linspace(14., 60., 921)                 #Extended redshift array
    t_long = (0.93e9)*(((1.+z_long)/7.)**(-1.5))        #Corresponding Hubble time array in years
    SFR_II_smooth = np.zeros((N,N,N,len(z)))
    SFR_III_smooth = np.zeros((N,N,N,len(z)))           #Initalize smoothed SFR arrays up to current z step
    for iz in range(899, z_iz, -1):                     #Then loop through redshift steps so far
        SFR_II_z = SFR_II_all[:,:,:,iz]                 #Isolate SFR(z) for both stellar populations
        SFR_III_z = SFR_III_all[:,:,:,iz]
        t_3 = t[iz] + t_life                            #Current Hubble time + PopIII stellar lifetime
        t_iz = np.argmin(np.abs(t_long-t_3))            #Find the corresponding index
        N_steps = iz - t_iz + 20                        #And the number of z steps which that timespan covers
        SFR_II_add = SFR_II_z/N_steps                   #Get SFR(z)/N_steps for both populations...
        SFR_III_add = SFR_III_z/N_steps                 #So that we can add these values to each of the following N_steps
        for i in range(0, N_steps):                     #Then loop through those steps to add the SFRs to...
            if iz-i < -0.5:                             #Near z = 15, don't let it smooth the SFR to end of z array (i.e. SFR[-1] = SFR(z=60))
                break
            SFR_II_smooth[:,:,:,iz-i] += SFR_II_add     #Add SFR_i_add to indices from z[iz] to z[iz+N_steps]
            SFR_III_smooth[:,:,:,iz-i] += SFR_III_add   #To both stellar populations to smooth SFR(z)
    return(SFR_III_smooth, SFR_II_smooth)

# Code that determines the Ly-alpha background everywhere at all z -----------------------------------------------------------------------
def Get_J_alpha(int N, double z_pres, np.ndarray[np.float64_t,ndim=1] z, np.ndarray[np.float64_t,ndim=4] SFRD_II_all, np.ndarray[np.float64_t,ndim=4] SFRD_III_all, np.ndarray d_scales, np.ndarray z_scales, np.ndarray[np.float64_t,ndim=1] E_H, np.ndarray[np.float64_t,ndim=1] f_rec, np.ndarray[np.float64_t,ndim=1] params):
    cdef int iN, N_iz, i, iz
    cdef double c, m_b, h, Omega_m, Omega_L, E_a, E_b, E_inf, h_p, const_z, E_i, z_smooth, d_smooth, dz_prime, N_cells, E_p, ep_b2, ep_b3, H_z, red, E_az, E_bz, E_infz
    cdef np.ndarray[np.float64_t,ndim=1] z_maxs, d_maxs
    cdef np.ndarray[np.float64_t,ndim=3] J_alpha_z, SFRD_II_0, SFRD_III_0
    cdef np.ndarray[np.float64_t,ndim=4] SFRD_II, SFRD_III, J_i

    c, m_b, h, Omega_m, Omega_L, E_a, E_b, E_inf, h_p = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]    #Necessary parameters for calculation
    const_z = (c * (1.+z_pres)**2.) / (4.*np.pi)                                            #Constant outside of integral for z_pres
    J_alpha_z = np.zeros((N, N, N))                     #Initialize this redshifts J_alpha(z) cube
    for iN in range(0, len(E_H)):                       #Now loop thru n=2-23 Hydrogen levels to sum contributions to J_alpha seen by all cells
        z_maxs, d_maxs = z_scales[iN], d_scales[iN]     #Array of redshifts & corresponding distance scales out to z_max(n) for this z_pres
        N_iz, E_i = len(z_maxs), E_H[iN]                #Number of shells we're smoothing over & photon energy for this energy level n
        J_i = np.zeros((N,N,N,N_iz))                                #Initialize J_alpha integrand array for all cells 
        SFRD_II, SFRD_III = np.zeros((N,N,N,N_iz)), np.zeros((N,N,N,N_iz))  #Initialize arrays of SFRDs which we will populate for each smoothing scale
        for i in range(0, N_iz):                                    #Now loop through and smooth SFRD cubes over scales in d_maxs (will integrate over)
            z_smooth, d_smooth = z_maxs[i], d_maxs[i]   #Current z & d smoothing scales (< z_max) for this enery level n at z_pres
            if i == 0:                                  #If this is the first shell, or if d(z_max) < 3.0 Mpc (meaning N_iz == 1)...
                dz_prime = np.abs(z_smooth - z_pres)    #Then dz' is just the difference from the current z step
            else:                                       #But, in cases with multiple shells out to z_max
                dz_prime = z_smooth - z_maxs[i-1]                   #Calculate the difference between z shells to get dz'
            iz, N_cells = np.argmin(np.abs(z-z_smooth)), d_smooth/3.                #Index of closest z step to current z-smoothing scale & number of cells over which we smooth
            SFRD_II_0, SFRD_III_0 = SFRD_II_all[:,:,:,iz], SFRD_III_all[:,:,:,iz]   #SFRDs at (approx.) z' for both populations
            if np.max(SFRD_III_0) < 1e-100 or dz_prime < 1e-100:                    #Skip this step if we're at z > z_onset
                continue
            if d_smooth >= 3.0:                                         #If the distance to z_max is >= cell size...
                SFRD_II[:,:,:,i] = smooth_cube(SFRD_II_0, N_cells, 1)   #Smooth the SFRD cube over that scale (average because SFRD is per volume)
                SFRD_III[:,:,:,i] = smooth_cube(SFRD_III_0, N_cells, 1) #Do this for both populations
            else:                                                       #Otherwise, just copy the SFRDs...
                SFRD_II[:,:,:,i] = np.copy(SFRD_II_0)   #As the photon has not yet had time to escape its host cell
                SFRD_III[:,:,:,i] = np.copy(SFRD_III_0) #So we don't smooth 
            red = (1.+z_smooth)/(1.+z_pres)             #Redshift factor from z' to z_pres
            E_p = E_i*red                               #Redshifted photon energy E'(n)
            if E_p >= E_a and E_p <= E_b:               #Now if photon energy is between E_alpha & E_beta...
                ep_b2 = 2902.91 * (E_p/E_inf)**(-0.86)  #Assign it this PopII SED (from Mittal & Kulkarni 2020)
                ep_b3 = 2691.91 * (E_p/E_inf)**(0.29)   #And this SED for PopIII (derived using same reference as M&K 2020 but for PopIII)
            elif E_p > E_b and E_p <= E_inf:            #If the photon energy is above E_beta but < ionizing
                ep_b2 = 1303.34 * (E_p/E_inf)**(-7.66)  #Assign it this PopII SED (M&K 2020)
                ep_b3 = 1155.98 * (E_p/E_inf)**(-6.89)  #And this PopIII SED (matching N_photons from Barkana & Loeb 2005)
            else:                                       #If it's not in either range...
                continue                #Skip it, it doesn't contribute
            ep_b2 *= h_p/1.60218e-12    #Convert SEDs from 1/eV --> 1/erg --> 1/Hz (s)
            ep_b3 *= h_p/1.60218e-12    #Do so for both stellar populations
            H_z = 2.333e-18 * (h/0.72) * np.sqrt(Omega_m*(1.+z_smooth)**3. + Omega_L)   #Hubble constant at z' (in s^-1)
            J_i[:,:,:,i] = (((ep_b2*2.146e-48*SFRD_II[:,:,:,i]) + (ep_b3*2.146e-48*SFRD_III[:,:,:,i])) / (m_b*H_z)) * dz_prime  #Now populate this integrand shell using the SFRD shells in cgs (g/s/cm^3)
        J_alpha_z += f_rec[iN] * np.sum(J_i, axis=3)    #After computing all shells, add up the contributions from all z' (integrate) & multiply by f_recycle(n)
    J_alpha_z *= const_z                                #Finally, after summing/integrating over all energy levels, multiply by constant outside summation
    J_alpha_z[J_alpha_z < 0.] = 0.      #Remove any negative values
    return J_alpha_z                    #And return the J_alpha(z_pres) seen by all cells

# Code that determines Lambda_ion & epsilon_X integrand summation terms, and updates X-ray optical depth ----------------------------------------------------------
def update_xray(np.ndarray[np.float64_t,ndim=1] z, int iz_i, np.ndarray[np.float64_t,ndim=1] E_X, np.ndarray[np.float64_t,ndim=1] dt_dz, np.ndarray[np.float64_t,ndim=1] n_bz, np.ndarray[np.float64_t,ndim=2] sig_H1z, np.ndarray[np.float64_t,ndim=2] sig_He1z, np.ndarray[np.float64_t,ndim=2] sigma_z, np.ndarray[np.float64_t,ndim=1] coeffs_H1, np.ndarray[np.float64_t,ndim=1] coeffs_He1, np.ndarray[np.float64_t,ndim=1] params):
    cdef int tau_dim, izz, N_steps
    cdef double c, x_e, f_H, f_He, f_heat_I, f_heat_II, Eth_H1, Eth_He1, dz, z_prime, z_2prime, dtdz, n_b, f_ion_H1, f_ion_He1, f_ion_sum
    cdef np.ndarray[np.float64_t,ndim=1] val_H1, val_He1, int_sum, ion_sum
    cdef np.ndarray[np.float64_t,ndim=2] tau_z, tau_int

    c, x_e, f_H, f_He, f_heat_I, f_heat_II, Eth_H1, Eth_He1, dz = params     #Assign the various parameter values
    #Start by calculating optical depth ---------------------------------------------------------------------------------
    z_prime, tau_dim = round(z[iz_i],2), len(z)-iz_i    #Current redshift (following notation in 21cmFast paper) & Number of steps > z'
    tau_z = np.zeros((tau_dim, len(E_X)))               #Initialize optical depth list for all z'' > z'
    for izz in range(iz_i, iz_i+len(sigma_z)):          #Now loop through all z'' > z' (looping over z^hat in 21cmFast) to populate tau_z with integrated values
        z_2prime = round(z[izz],2)                      #Current lookback redshift for z'
        N_steps = izz - iz_i                            #Number of redshift steps between z' & z''
        if N_steps == 0:                                #If this number is zero (z' = z''), then "integrating" would just be computing integrand once at z'
            tau_z[N_steps] = c*dt_dz[iz_i]*n_bz[iz_i]*sigma_z[0]*dz     #Like so, where sigma_z[0] is sigma(z',nu)
        else:                                           #Otherwise, we'll need to integrate over all z^hat
            tau_int = np.zeros((N_steps, len(E_X)))     #So initialize integrand array which we'll sum over N_steps to get tau(nu) for this z'-z'' pair 
            for i in range(0, N_steps):                 #Now loop through all redshifts between z' & z'' to populate integrand array
                dtdz, n_b = dt_dz[iz_i+i], n_bz[iz_i+i] #Values of dt/dz & n_bz at z^hat = z'+i until z^hat = z'+N_steps = z''
                tau_int[i] = c*dtdz*n_b*sigma_z[i]*dz   #Populate integrand array for this z^hat, sigma~(z',nu^hat) = ith index of all sigma(z') (still f(nu))
            tau_z[N_steps] = np.sum(tau_int, axis=0)    #Once the integrand array is full after N_steps, integrate by summing over, leaving frequency dependence
    #Then calculate integral summation term for getting Epsilon_X --------------------------------------------------------
    val_H1 = ((E_X-Eth_H1)*1.60218e-12)*f_heat_I*f_H*(1.-x_e)*sig_H1z[0]
    val_He1 = ((E_X-Eth_He1)*1.60218e-12)*f_heat_I*f_He*(1.-x_e)*sig_He1z[0]    #Calculate each index of the summation
    int_sum = val_H1 + val_He1                                                  #Then sum the values to get Sum_i(nu)
    #And finally calculate the summation term for getting Lambda_ion -----------------------------------------------------
    f_ion_H1 = coeffs_H1[0] * ((1.-(x_e**coeffs_H1[1]))**coeffs_H1[2])          #Calculate f_ion(z) for x_e(z)
    f_ion_He1 = coeffs_He1[0] * ((1.-(x_e**coeffs_He1[1]))**coeffs_He1[2])      #From both HI and HeI (ignore HeII)
    f_ion_sum = (f_ion_H1/Eth_H1) + (f_ion_He1/Eth_He1)                                 #Summation in F_i term of integrand summation
    val_H1 = (((E_X-Eth_H1)*1.60218e-12) * f_ion_sum + 1.) *f_H*(1.-x_e)*sig_H1z[0]     #Then calculate value of each summation term
    val_He1 = (((E_X-Eth_He1)*1.60218e-12) * f_ion_sum + 1.) *f_He*(1.-x_e)*sig_He1z[0] #Multiply E_X term by conversion factor for ev --> erg
    ion_sum = val_H1 + val_He1                          #Then sum the values
    return(tau_z, ion_sum, int_sum)

# Code that determines the X-ray photon heating rate for T_k ------------------------------------------------------------------------------------------------------
def Get_ep_X(int N, int iz_0, int iz_i, np.ndarray[np.float64_t,ndim=1] z, np.ndarray[np.float64_t,ndim=4] dNx_dz_all, np.ndarray[np.float64_t,ndim=2] tau_X, np.ndarray[np.float64_t,ndim=1] dists, np.ndarray[np.float64_t,ndim=1] sum_z, np.ndarray[np.float64_t,ndim=1] nu_X, np.ndarray[np.float64_t,ndim=1] dphi_C, double alpha_S, double nu_0, np.ndarray[np.float64_t,ndim=1] dists_p, np.ndarray[np.float64_t,ndim=1] sum_ion):
    cdef int index, iz, i
    cdef double d_smooth, N_cells, red, SAz
    cdef np.ndarray[np.float64_t,ndim=1] f_nu, tau_Xz, J_nu_z
    cdef np.ndarray[np.float64_t,ndim=3] dNx_0, dNx_dz, nu_int, nu_ion
    cdef np.ndarray[np.float64_t,ndim=4] z_int

    index, d_nu = 1, nu_X[1]-nu_X[0]        #Initialize index to assign values & spacing in nu_X array
    z_int = np.zeros((N,N,N,len(nu_X)))             #Initialize 4D array, 3D cells(x) + 1D E_X(nu), integrated over current z = z' but not yet over nu
    for iz in range(iz_i+1, min(iz_0, iz_i+100)):           #Begin looping through time up to iz_0 to get contributions from each z''
        d_smooth, tau_Xz = dists[index], tau_X[index]       #Smoothing distance from z'-z'' and corresponding optical depth of X-rays for all z_higher
        dNx_0, N_cells = dNx_dz_all[:,:,:,iz], d_smooth/3.  #X-ray emission rate at z'' and # of cells over which we'll smooth
        start = time.time()
        if d_smooth >= 3.0:                             #Now calculate integral values
            dNx_dz = smooth_cube(dNx_0, N_cells, 0)     #If d(z',z'') > 1 cell, smooth it over that distance
        else:                                           #Otherwise, just copy values from present z as X-rays haven't left cell
            dNx_dz = np.copy(dNx_0)
        red = ((1.+z[iz])/(1.+z[iz_i]))**(-alpha_S-1.)  #Redshift term in dPhix_dz'' (from z'' to z')
        SAz = 4.*np.pi*((dists_p[index]*3.0857e24)**2.) #Surface area of sphere (proper cm^2)
        f_nu = (dphi_C*red*np.exp(-tau_Xz))/SAz         #All constants * f(nu) terms after dNx/dz'' in dPhix/dz'' integral of epsilon_X equation
        z_int += dNx_dz[:,:,:,np.newaxis]*f_nu[np.newaxis,np.newaxis,np.newaxis,:]*0.05         #Record values in integral array for this nu
        index += 1                                      #Iterate index for next z''
    J_nu_z = np.mean(np.mean(np.mean(z_int, axis=0), axis=0), axis=0)   #Populate J_X(nu) array with averages across x/y/z dimensions 
    nu_int = d_nu*np.sum(sum_z*z_int, axis=3)           #Then integrate by summing z_int * summation term * d_nu to get ep_X
    nu_ion = d_nu*np.sum(sum_ion*z_int, axis=3)         #And integrate * ionization summation term * d_nu to get dx_e/dz (next step)
    del z_int                                           #Delete array for memory
    return(nu_int, J_nu_z, nu_ion)

# Code that solves T_K ODEs for all cells at each z step ----------------------------------------------------------------------------------------------------------
def Get_T_K(np.ndarray[np.float64_t,ndim=4] r, double z_pres, np.ndarray[np.float64_t,ndim=1] vals, np.ndarray[np.float64_t,ndim=3] ep_X0, np.ndarray[np.float64_t,ndim=3] ion_X0, np.ndarray[np.float64_t,ndim=3] chi, np.ndarray[np.float64_t,ndim=3] phi): 
    cdef double dtdz, psi, T_gamma, f_He, k_B
    cdef np.ndarray[np.float64_t,ndim=3] x_e, T_k, dxe_dz, dTk_dz, ep_C, ep_X

    x_e, T_k, dtdz, psi = r[0], r[1], vals[0], vals[1]  #Current values of x_e and T_K, dt/dz, and the parameters used in calculating x_e & in T_k
    T_gamma, f_He, k_B = vals[2], vals[3], vals[4]      #CMB temperature, # fraction of He, X-ray heating rate of this cell
    dxe_dz = dtdz * (ion_X0 - chi*x_e**2.)              #Now calculate dx_e/dz' and dT_k/dz'
    ep_C = psi*(x_e/(1.+f_He+x_e))*(T_gamma-T_k)        #Compton heating rate of all cells
    ep_X = (2.*ep_X0)/(3.*k_B*(1.+x_e))                 #Fraction outside summation * X-ray heating rate
    dTk_dz = dtdz*(ep_C+ep_X) + (2.*T_k)/(1.+z_pres) + ((2.*T_k)/3.)*phi - (T_k*dxe_dz)/(1.+x_e)
    return np.array([dxe_dz, dTk_dz])                   #Return updated ODE solutions

# Code that iteratively determines S_alpha --> x_alpha --> T_spin until T_spin converges for all cells --------------------------------------------------------------------
def Get_T_S(int N, double z_pres, np.ndarray[np.float64_t,ndim=3] T_K, np.ndarray[np.float64_t,ndim=3] J_a_all, np.ndarray[np.float64_t,ndim=3] xi, np.ndarray[np.float64_t,ndim=3] x_c, np.ndarray[np.float64_t,ndim=1] vals):
    cdef int i
    cdef double T_gamma, lambda_lya, T_star, gamma, A_10
    cdef np.ndarray[np.float64_t,ndim=3] T_S_all, T_c_eff, S_alpha_a, S_alpha_b, S_alpha, x_alpha, T_spin, perc_diff

    T_gamma, lambda_lya, T_star, gamma, A_10 = vals[0], vals[1], vals[2], vals[3], vals[4]  #Values needed for T_S calculation
    T_S_all = np.full((N,N,N), fill_value=T_gamma)                                          #Populate a 3D array with T_spin initial guess value (T_CMB) for this z
    for i in range(0, 100):                                                                 #Begin looping through & iteratively updating values -- Should converge well before i = 100
        T_c_eff = 1./((1./T_K) + 0.405535 * (1./T_K) * ((1./T_S_all) - (1./T_K)))           #Effective color temperature of all cells, to be used in T_S calculation
        S_alpha_a = 1. - 0.0631789*(1./T_K) + 0.115995*((1./T_K)**2.) - 0.401403*(1./T_K)*(1./T_S_all) + 0.336463*(1./T_S_all)*((1./T_K)**2.)
        S_alpha_b = (1. + 2.98394*xi + 1.53583*xi**2. + 3.85289*xi**3.)**-1.                #Terms for S_alpha from Hirata 2006
        S_alpha = S_alpha_a * S_alpha_b                                                                 #Calculate S_alpha for all cells
        x_alpha = (8.*np.pi*(lambda_lya**2.)*gamma*T_star*S_alpha*J_a_all)/(9.*A_10*T_gamma)            #Calculate W-F coupling coefficients as well
        T_spin = (((T_gamma**-1.) + x_alpha*(T_c_eff**-1.) + x_c*(T_K**-1.)) / (1.+x_alpha+x_c))**-1.   #Finally, calculate spin temperature for all cells
        perc_diff = np.abs(T_S_all - T_spin)/T_S_all    #Calculate percent difference between latest T_spin & previous values
        if np.max(perc_diff) < 0.01:                    #If values changed by < 1%, it has converged
            return T_spin               #Return the spin temperature & quit calculation
        else:
            T_S_all = T_spin            #Otherwise, update T_S_all & continue looping

# Code to calculate the power spectrum -----------------------------------------------------------------------------------------------------------------------
def Power(double Lx, np.ndarray[np.float64_t,ndim=3] field1, np.ndarray[np.float64_t,ndim=3] field2, int Nbins):
    cdef int Nx, i
    cdef double dx, V_pix
    cdef np.ndarray[np.float64_t,ndim=1] k1d, karr, P_sum, P_count, PS, k_cen
    cdef np.ndarray[np.float64_t,ndim=3] kx, ky, kz, fftop, kmag
    cdef np.ndarray[np.complex128_t,ndim=3] cube1, cube2

    Nx = field1.shape[0]                        #Number of cells/side in halo map
    dx = Lx/Nx                                  #Width of each cell
    V_pix = dx**3.
    cube1, cube2 = np.fft.fftn(field1), np.fft.fftn(field2)
    cube1, cube2 = np.fft.fftshift(cube1), np.fft.fftshift(cube2)
    fftop = np.real(cube1*np.conjugate(cube2))  #Definition of power spectrum -- ~delta(k) x ~delta*(k)
    k1d = np.fft.fftfreq(Nx, d=Lx/(2*np.pi*Nx)) #Initialize 1D array of frequency bins
    k1d = np.fft.fftshift(k1d)                  #Shift zero-frequency to center
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d)     #Make 3D grid of frequencies
    kmag = np.sqrt(kx**2+ky**2+kz**2)           #And calculate the k-magnitude for each cell
    karr = np.logspace(np.log10(2*np.pi/Lx),np.log10(kmag.max()),Nbins+1)   #Initialize logspace k-bin array
    P_sum, P_count = np.zeros(Nbins), np.zeros(Nbins)                       #Initialize sum of power on each scale & array for # within each bin
    for i in range(Nbins-1):                                        #Now loop through each distance scale
        indices = np.bitwise_and(karr[i]<=kmag, kmag<karr[i+1])     #Take bitwise AND of cells within k-bin
        P_sum[i] = np.sum(fftop[indices])                           #Add relevent values from convolved FFT cube
        P_count[i] = np.sum(indices)                   #And add number of cells that fell within this range
    PS = (1./(Lx**3.)) * V_pix**2. * (P_sum/P_count)   #Calculate power
    PS = np.nan_to_num(PS)
    k_cen = np.sqrt(karr[:-1]*karr[1:])                #Geometric means of each k-bin
    return PS, k_cen, P_count

# Main code that calls the other functions ------------------------------------------------------------------------------------------------------------
def Get_21cm(int sim, str special, np.ndarray[np.int_t,ndim=1] get, int overwrite=0, int N_star=4, double star_m0=35.69): 
    cdef int N_side, iz, ii, ij, ik, izz, C_min_i, N_bins, N_slice, i, slice_i, iz_a, iz_b, iz_0, iz_1
    cdef str save_path
    cdef double H_0, Omega_m, c, m_H, h_p, H_0_s, h, Omega_L, m_b, V, n_b0, T_star, x_HI, n_H0, lambda_lya, gamma, A_10, k_B, E_alpha, E_beta, E_inf, z_pres, h_P, zeta_X, nu_0, alpha_S, dz, dx, sigma_T, Omega_b, f_H, f_He, U_CMB, m_e, f_heat_I, f_heat_II, Eth_H1, Eth_He1, C_min, z_dec, T_k0, x_e0, dtdz, x_e, z_prime, red, x_e_prime, alpha_A, U_gamma, T_gamma, psi, H_z, nu_21, dz_b, dz_0, dz_1
    cdef list sig_H1z, sig_He1z, sig_z
    cdef np.ndarray[np.int_t,ndim=1] n_H
    cdef np.ndarray[np.float64_t,ndim=1] z, t, Dz_dot, Dz, dt_dz, f_recycle, E_H, J_params, J_alpha_avg, J_alpha_sd, E_X, nu_X, dphi_Const, x_H1, y_H1, sig_H1, x_He1, y_He1, sig_He1, sig, alpha_Az, n_b_z, T_Kz, C_all, H1, He1, coeffs_H1, coeffs_He1, x_e_avg, x_e_sd, T_k_avg, T_k_sd, E_prime, params, ion_sum, int_sum, values, T_S_avg, T_S_sd, T_k_params, z_LC, H, z_dists, all_dz, z_fine
    cdef np.ndarray[np.float64_t,ndim=2] J_X_avg, tau_X_all, Pk_all, k_all, nk_all
    cdef np.ndarray[np.float64_t,ndim=3] all_delta, n_bz, chi, phi, x_ez, T_K, J_a, n_Hz, tau_GP, xi, log_Tk, kappa_HH, kappa_eH, n_e, kappa_pH, x_c, T_spin, tau_21, T_b, LC,
    cdef np.ndarray[np.float64_t,ndim=4] SFRD_all, SFR_III_0, SFR_II_0, SFR_III_all, SFR_II_all, SFRD_III_all, SFRD_II_all, J_alpha_all, dNx_dz_all, ep_X_all, ion_all, x_e_all, T_k_all, r, k1, k2, k3, k4, T_S_all, T_b_all

    N_side, z_0 = 64, 1                     #Cells/side of the sim volume, and index to which we calculate things (len(z)-z_0)
    if sim == 0:
        save_path = './Results/Box_' + str(N_side) + special + '/'      #Path to save results for Paper 2 Fiducial method...
    elif sim == 1:
        save_path = './Results/Integral_' + str(N_side) + special + '/' #For the HMF integral method...
        z_0 = 2
    try:
        os.mkdir(save_path + '21cm/')       #Create 21cm results directory if needed 
    except OSError:
        print(OSError)

    coeffs = np.load('./Schaerer_PopIII_Fit_Coeffs.npy', allow_pickle=True) #Fitting coefficients for the PopIII stellar lifetime from Schaerer 2002
    coeffs_life, fit_life = coeffs[0], 0.                                   #Initialize fit value for PopIII lifetime
    for i in range(0, len(coeffs_life)):
        fit_life += coeffs_life[i]*np.log10(star_m0)**float(len(coeffs_life)-i-1)   #Calculate fit value
    print('(Log) Lifetime = ', fit_life)
    fit_life = 10.**(fit_life)              #Raise fit to 10^fit 

    z = np.linspace(15., 60., 901)          #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))      #Hubble time array in years
    all_delta = np.load('./Overdensity_Field_64_192.npy')   #Overdensities of all cells
    Dz_dot = np.load('./Dz_dot_dz.npy')     #Derivative of growth factor, D(z)
    Dz = np.load('./Dz_15.npy')             #Growth factor at each redshift
    dt_dz = np.load('./dt_dz.npy')*3.154e7  #Change in Hubble time/Change in redshift for each z (sec)

    H_0, Omega_m, c, m_H, h_p = 67., 0.32, 29979245800., 1.67356e-24, 6.6261e-27    #Hubble constant (km/s/Mpc), cosmo matter density param, light speed (cm/s), H atom mass (g), cgs Planck const.
    H_0_s, h, Omega_L, m_b, V = H_0/3.0857e19, H_0/100., 1.-Omega_m, 1.22*m_H, 27.  #Hubble const in 1/s, little h, Lambda cosmo density paramter, baryon mass, cell volume (Mpc^3)
    n_b0, T_star, n_H0, nu_21 = 2.47113e-7, 0.06813, 1.8533e-7, 1.42041e9           #Present baryon # density (cm^-3), temp. of H(1-0) transition (K, for ratio of excited states), H # density (cm^-3), 21-cm photon frequency (Hz)
    lambda_lya, gamma, A_10, k_B, x_HI = 1.21567e-5, 5e7, 2.85e-15, 1.3807e-16, 1.  #Lyman-a wavelength (cm), HWHM of Ly-a resonance (Hz), intrinsic width of hyperfine level (s^-1), Boltzmann const (erg/K), & neutral H fraction

    if get[0] == 1:
        print('Getting J_alpha_continuum')
        try:
            SFRD_II_all, SFRD_III_all = np.load(save_path + '21cm/SFRD_II_Smoothed_all.npy'), np.load(save_path + '21cm/SFRD_III_Smoothed_all.npy') #Try loading in smoothed total SFRD
        except:
            SFR_III_0, SFR_II_0 = np.load(save_path + 'SFR_III_all.npy'), np.load(save_path + 'SFR_II_all.npy') #PopII & PopIII SFR(z) cubes
            SFR_III_all, SFR_II_all = Smooth_SFRs(N_side, -1, z, t, SFR_III_0, SFR_II_0, fit_life)              #Smooth SFRs over PopIII stellar lifetime (fiducial=3 Myr)
            SFRD_III_all, SFRD_II_all = SFR_III_all/V, SFR_II_all/V                     #Divide by cell volume to get into SFRDs
            np.save(save_path + '21cm/SFRD_II_Smoothed_all.npy', SFRD_II_all)
            np.save(save_path + '21cm/SFRD_III_Smoothed_all.npy', SFRD_III_all)         #Save each for future reference
            del SFR_III_0, SFR_II_0, SFR_III_all, SFR_II_all                            #Delete very large arrays for efficiency
        z_smooth_all = np.load('./J_alpha_Smoothing_Redshifts.npy', allow_pickle=True)  #Individual smoothing redshifts for each z_max(n,z)
        d_smooth_all = np.load('./J_alpha_Smoothing_Scales.npy', allow_pickle=True)     #And corresponding distance scales 
        f_recycle = np.load('./Ly_alpha_f_rec.npy')                                     #Chance that a photon at level n cascades to Ly-alpha
        E_alpha, E_beta, E_inf = 10.2, 12.09, 13.6                      #Energies of Ly-alpha, -beta, and ionization threshold (eV)
        n_H = np.arange(2, 24)                                          #Relevant energy levels of hydrogen 
        E_H = 13.6 * (1. - n_H**-2.)                                    #Photon energies from each H level
        J_params = np.array([c, m_b, h, Omega_m, Omega_L, E_alpha, E_beta, E_inf, h_p]) #Values needed for J_alpha(z) calculation
        J_alpha_avg, J_alpha_sd = np.zeros(len(z)), np.zeros(len(z))    #Initialize average J_alpha(z) array & Std. Dev. array
        J_alpha_all = np.zeros((N_side,N_side,N_side,len(z)))           #Initialize 4D array of all J_alpha evolutions
        for iz in range(len(z)-z_0, -1, -1):                            #Begin stepping through time to determine J_alpha(z)
            z_pres = round(z[iz],2)                                     #Present redshift
            z_scales, d_scales = z_smooth_all[iz], d_smooth_all[iz]     #Arrays of smoothing distance scales & corresponding redshifts
            if np.sum(SFRD_III_all[:,:,:,iz:len(z)]) < 1e-200:
                print(z_pres)
                continue
            J_alpha_all[:,:,:,iz] = Get_J_alpha(N_side, z_pres, z, SFRD_II_all, SFRD_III_all, d_scales, z_scales, E_H, f_recycle, J_params)  #Calculate J_alpha(z_pres) for all cells
            J_alpha_avg[iz], J_alpha_sd[iz] = np.mean(J_alpha_all[:,:,:,iz]), np.std(J_alpha_all[:,:,:,iz])                    #Also get average & std.dev. at z_pres
            print(z_pres, np.min(J_alpha_all[:,:,:,iz]), J_alpha_avg[iz], np.max(J_alpha_all[:,:,:,iz]))
            if iz % 10 == 0 and overwrite == 1:
                np.save(save_path + '21cm/J_alpha_all.npy', J_alpha_all)
                np.save(save_path + '21cm/J_alpha_avg_SD.npy', np.column_stack((J_alpha_avg, J_alpha_sd)))

    if get[1] == 1:
        print('Getting X-ray heating, and T_k & x_e evolution')         #Code combining getting epsilon_X --> update T_k/x_e --> update sigma_i --> update summation terms & tau_X
        h_P, zeta_X, alpha_S = 4.1357e-15, 1e57/1.989e33, 1.            #Planck const (eV*s), ionizing efficiency (g^-1) and spectral index (0.5=miniquasar, 1 =SNe remnants, 1.5=starburst)
        E_X, nu_0 = 1000. * np.linspace(0.1,30.,500), 200./h_P          #X-ray photon energy range & lowest freq. Xray to escape into IGM in eV /h_P to convert to Hz
        nu_X, dz, sigma_T, Omega_b = E_X/h_P, -0.05, 6.6525e-25, 0.049      #X-ray photon frequencies, dz of z array, Thomson scattering cross-sec (cm^2), cosmo baryon density parameter
        f_H, f_He, U_CMB, m_e = 0.9231, 0.0769, 4.17e-13, 9.109e-28         #H & He baryon N fractions, current CMB energy density (ergs/cm^3), electron mass (g) 
        f_heat_I, f_heat_II, Eth_H1, Eth_He1 = 0.979, 0.1352, 13.6, 24.587  #f_heat values (assumed constant) for HI & HeI & Ionization threshold energies (eV)
        alpha_Az = np.load('./Case_A_Recombinations.npy')                   #Case-A recombination coeff (cm^3/s) as f(T_K), fit from Pequignot+ 1991
        n_b_z = np.load('./Avg_n_Baryon_z.npy')                 #Average baryon number density by z for my overdensity field
        T_Kz = np.logspace(-1., 5., 1000)                       #Temperature range over which Case-A coefficients are calculated
        C_all = 27.466 * np.exp(-0.114*z + 0.001328*z**2.)      #Clumping factor as f(z) (fit used in 21cmFast for 8<z<40)
        C_min = min(C_all)                                      #Find minimum C value
        C_min_i = int(np.argmin(np.abs(C_all-C_min)))           #Find it's index in overall array
        C_all[C_min_i:] = C_min                                 #Set every z > z(C=min) to be C = C_min
        z_dec = 137. * ((Omega_b*h**2.)/0.022)**0.4 - 1.                #Redshift at which CMB photons decouple from gas temperature
        T_k0, x_e0 = 2.73*(1.+z_dec)*(61./(1.+z_dec))**2., 0.0002       #Initial guess for T_k0 given adiabatic cooling & of x_e0 from Bera+2020
        dphi_Const = (alpha_S/nu_0)*((nu_X/nu_0)**(-alpha_S-1.))        #Array of values f(nu) used in calculating dPhix_dz''
        dists_z = np.load('./All_z_Distances.npy',allow_pickle=True)        #Array of comoving distances from z-z' for smoothing dNx_dz
        SFRD_all = V * (np.add(np.load(save_path + '21cm/SFRD_II_Smoothed_all.npy'), np.load(save_path + '21cm/SFRD_III_Smoothed_all.npy')))   #Load in all smoothed total SFRDs(x,z), times V_cell = SFR(x,z)
        dNx_dz_all = zeta_X * (SFRD_all*1.989e33/3.154e7)                   #Smoothed SFR(x,z) converted to g/s times ionizing efficiency = emission rate (s^-1)
        del SFRD_all    #Delete SFRD array to save memory
        os.remove(save_path + '21cm/SFRD_II_Smoothed_all.npy')
        os.remove(save_path + '21cm/SFRD_III_Smoothed_all.npy')
        H1 = np.array([0.4298, 5.475e-14, 32.88, 2.963, 0., 0., 0.])            #Fitting parameters for calculating photoionizing cross-section (cm^2) 
        He1 = np.array([13.61, 9.492e-16, 1.469, 3.188, 2.039, 0.4434, 2.136])  #Parameters from left to right: E_0 (eV), sigma_0 (cm^2), y_a, P, y_w, y_0, y_1 from Verner+1996 (Table 1)
        coeffs_H1 = np.array([0.3908, 0.4092, 1.7592])                          #Fitting parameters for secondary ionizations as f(x_e)
        coeffs_He1 = np.array([0.0554, 0.4614, 1.666])                          #Following EQ 2 of Shull & VanSteenberg (1985), in order of C, a, b
        ############################################################################################################
        ep_X_all = np.zeros((N_side,N_side,N_side,len(z)))      #Initialize heating rate per baryon from X-rays
        ion_all = np.zeros((N_side,N_side,N_side,len(z)))       #Initialize ionization rate per baryon from X-rays
        J_X_avg = np.zeros((len(z),len(nu_X)))                  #Also initialize average X-ray background intensity, J_X(nu,z)
        x_e_all = np.zeros((N_side,N_side,N_side,len(z)))       #Initialize x_e(z) and T_k(z) arrays for all cells
        T_k_all = np.zeros((N_side,N_side,N_side,len(z)))       #We will be using the method described in Mesinger+2010 (21cmFast)
        x_e_avg, x_e_sd = np.zeros(len(z)), np.zeros(len(z))    #And initialize average/st. dev. arrays
        T_k_avg, T_k_sd = np.zeros(len(z)), np.zeros(len(z))    #For both T_K & x_e
        e_X_avg, e_X_sd = np.zeros(len(z)), np.zeros(len(z))    #And epsilon_X
        x_e_all[:,:,:,len(z)-1] = np.full((N_side,N_side,N_side), fill_value=x_e0)  #Fill step before initial z of T_K & x_e arrays...
        T_k_all[:,:,:,len(z)-1] = np.full((N_side,N_side,N_side), fill_value=T_k0)  #...with their initial guesses & update avgs/SDs
        x_e_avg[len(z)-1], x_e_sd[len(z)-1] = np.mean(x_e_all[:,:,:,len(z)-1]), np.std(x_e_all[:,:,:,len(z)-1])
        T_k_avg[len(z)-1], T_k_sd[len(z)-1] = np.mean(T_k_all[:,:,:,len(z)-1]), np.std(T_k_all[:,:,:,len(z)-1])
        r = np.array((x_e_all[:,:,:,len(z)-1], T_k_all[:,:,:,len(z)-1]))          #And finally, initialize R-K array
        runtime = np.zeros(len(z))
        start = time.time()
        for iz in range(len(z)-1, -1, -1):                  #Begin stepping through time to get epsilon_X(x,z)
            z_pres, dtdz = round(z[iz],2), dt_dz[iz]        #Present redshift & Hubble time gradient (sec)
            if np.mean(dNx_dz_all[:,:,:,iz]) > 1e-100:      #Update photoionization & X-ray background once SF begins
                # ---- Start with updating photoionization cross-sections ------------------------------------
                sig_H1z, sig_He1z, sig_z = [], [], []       #Initialize sigma(E,z') for all z' > z_pres (for both HI & HeI, and sigma~)
                x_e = np.mean(x_e_all[:,:,:,iz+1])          #Average electron fraction of previous step for optical depth
                for izz in range(iz, len(z)-z_0+1):         #Loop through z' > z_pres (up to onset of SF) to get all sigma(E) at each
                    z_prime = round(z[izz],2)               #Current z' for z_pres
                    red = (1.+z_prime)/(1.+z_pres)          #Redshifting from z' - z_pres
                    E_prime = E_X * red                     #Redshifted x-ray photon energies
                    x_H1 = (E_prime/H1[0]) - H1[5]              #Parameter 1 for sigma fit (for H1)
                    y_H1 = np.sqrt(x_H1**2. + H1[6]**2.)        #Parameter 2 "" -- Then calculate sigma(E,z') for this z_pres using fitting parameters
                    sig_H1 = H1[1] * ((x_H1-1.)**2. + H1[4]**2.) * y_H1**(0.5*H1[3]-5.5) * (1.+np.sqrt(y_H1/H1[2]))**(-H1[3])
                    x_He1 = (E_prime/He1[0]) - He1[5]           #Parameter 1 for sigma fit (for He1)
                    y_He1 = np.sqrt(x_He1**2. + He1[6]**2.)     #Parameter 2 "" -- Then calculate sigma(E,z') for this z_pres using fitting parameters
                    sig_He1 = He1[1] * ((x_He1-1.)**2. + He1[4]**2.) * y_He1**(0.5*He1[3]-5.5) * (1.+np.sqrt(y_He1/He1[2]))**(-He1[3])
                    x_e_prime = np.mean(x_e_all[:,:,:,izz])     #Mean electron fraction at z''
                    sig = f_H*(1.-x_e_prime)*sig_H1 + f_He*(1.-x_e_prime)*sig_He1  #Calculate sigma~ for all nu at this z'
                    sig_H1z.append(sig_H1)                      #Append arrays with this z' sigma(E) values at z_pres
                    sig_He1z.append(sig_He1)                    #For both HI and HeI
                    sig_z.append(sig)                           #And also the sigma~ array
                params = np.array([c, x_e, f_H, f_He, f_heat_I, f_heat_II, Eth_H1, Eth_He1, dz])     #Now use updated cross-sections to update integrand summation terms
                tau_X_all, ion_sum, int_sum = update_xray(z, iz, E_X, dt_dz, n_b_z, np.array(sig_H1z), np.array(sig_He1z), np.array(sig_z), coeffs_H1, coeffs_He1, params) #Use updated cross-sections to update integral summations & optical depth
                # ---- Now to get X-ray heating rate & background intensity ---------------------------------
                ep_X_all[:,:,:,iz], J_X_avg[iz], ion_all[:,:,:,iz] = Get_ep_X(N_side, len(z)-z_0, iz, z, dNx_dz_all, tau_X_all, dists_z[iz], int_sum, nu_X, dphi_Const, alpha_S, nu_0, dists_z[iz]/(1.+z_pres), ion_sum)
            # ---- And finally, update T_K & x_e of all cells for this z --------------------------------
            n_bz = n_b0*(1.+z_pres)**3. * (1.+all_delta*(Dz[iz]/Dz[0])) #Redshifted baryon density of all cells
            alpha_A = alpha_Az[np.argmin(np.abs(T_Kz-np.mean(r[1])))]   #Case-A Recombo. Coeff. for the average T_K
            U_gamma = U_CMB * (1.+z_pres)**4.                           #Redshifted CMB photon density
            T_gamma = 2.725 * (1.+z_pres)                               #And CMB temperature
            chi = alpha_A * C_all[iz] * f_H * n_bz                      #Parameter defined for x_e ODE
            psi = (8.*sigma_T*U_gamma)/(3.*m_e*c)                       #& Parameter 1 of 2 for T_k ODE...
            phi = Dz_dot[iz]/(Dz[0]/all_delta + Dz[iz])                         #And 2 of 2 for T_k ODE I defined
            values = np.array([dtdz, psi, T_gamma, f_He, k_B])                             #Assemble values array for this cell at this z
            k1 = dz*Get_T_K(r, z_pres, values, ep_X_all[:,:,:,iz], ion_all[:,:,:,iz], chi, phi)
            k2 = dz*Get_T_K(r+0.5*k1, z_pres+0.5*dz, values, ep_X_all[:,:,:,iz], ion_all[:,:,:,iz], chi, phi) #Calculate 4th order Runge-Kutta functions
            k3 = dz*Get_T_K(r+0.5*k2, z_pres+0.5*dz, values, ep_X_all[:,:,:,iz], ion_all[:,:,:,iz], chi, phi) #To update & replace dx_e/dz & dT_k/dz
            k4 = dz*Get_T_K(r+k3, z_pres+dz, values, ep_X_all[:,:,:,iz], ion_all[:,:,:,iz], chi, phi)
            r += (k1 + 2.*k2 + 2.*k3 + k4)/6.                           #Update r vector with new dx_e/dz & dT_k/dz values for next step
            x_e_all[:,:,:,iz], T_k_all[:,:,:,iz] = r[0], r[1]           #And update x_e & T_k arrays, including their averages/SDs
            x_e_avg[iz], x_e_sd[iz] = np.mean(x_e_all[:,:,:,iz]), np.std(x_e_all[:,:,:,iz])
            T_k_avg[iz], T_k_sd[iz] = np.mean(T_k_all[:,:,:,iz]), np.std(T_k_all[:,:,:,iz])
            e_X_avg[iz], e_X_sd[iz] = np.mean(ep_X_all[:,:,:,iz]), np.std(ep_X_all[:,:,:,iz])
            runtime[iz] = time.time() - start
            print(z_pres)
            print('Epsilon_X: ', np.min(ep_X_all), np.mean(ep_X_all), np.max(ep_X_all))
            print('J_X_avg: ', J_X_avg[iz][::100])
            print('x_e & T_K: ', x_e_avg[iz], T_k_avg[iz])
            if iz % 50 == 0 and overwrite == 1:
                np.save(save_path + '21cm/Epsilon_X_all' + special + '.npy', ep_X_all)
                np.save(save_path + '21cm/Epsilon_X_Avg' + special + '.npy', np.column_stack((e_X_avg, e_X_sd)))
                np.save(save_path + '21cm/J_X_avg' + special + '.npy', J_X_avg)
                np.save(save_path + '21cm/Xray_ion' + special + '.npy', ion_all)
                np.save(save_path + '21cm/T_k_all' + special + '.npy', T_k_all)
                np.save(save_path + '21cm/x_e_all' + special + '.npy', x_e_all)
                np.save(save_path + '21cm/T_k_Avg' + special + '.npy', np.column_stack((T_k_avg, T_k_sd)))
                np.save(save_path + '21cm/x_e_Avg' + special + '.npy', np.column_stack((x_e_avg, x_e_sd)))
                np.save(save_path + '21cm/Runtime' + special + '.npy', runtime)
        os.remove(save_path + '21cm/Xray_ion' + special + '.npy')   #This is only being saved in case the job is killed, delete from OSC for memory once completed

    if get[2] == 1:
        print('Calculating Spin Temperatures')
        T_k_all = np.load(save_path + '21cm/T_k_all' + special + '.npy')    #The T_K(z) evolution of all cells
        x_e_all = np.load(save_path + '21cm/x_e_all' + special + '.npy')    #And the electron fraction, x_e, evolution of all cells
        J_alpha_all = np.load(save_path + '21cm/J_alpha_all.npy')           #And their Ly-a background intensities
        T_S_all = np.zeros((N_side, N_side ,N_side ,len(z)))        #Initialize T_S(x,z) for all cells
        T_S_avg, T_S_sd = np.zeros(len(z)), np.zeros(len(z))        #Also initialize mean/st. dev. of T_b(z) arrays
        for iz in range(len(z)-1, -1, -1):                          #Loop through time to calculate T_s(z) for all cells
            z_pres = round(z[iz],2)                                 #Current redshift
            T_gamma, x_ez = 2.726 * (1.+z_pres), x_e_all[:,:,:,iz]  #Current CMB temperature -- Initial guess for T_S(z) & current x_e
            T_K, J_a = T_k_all[:,:,:,iz], J_alpha_all[:,:,:,iz]             #Current T_k and J_alpha for all cells
            n_bz = n_b0 * (1.+z_pres)**3. * (1.+all_delta*(Dz[iz]/Dz[0]))   #Redshifted baryon number density
            n_Hz = n_H0 * (1.+z_pres)**3. * (1.+all_delta*(Dz[iz]/Dz[0]))               #Redshifted Hydrogen number density
            H_z = 2.333e-18 * (h/0.72) * np.sqrt(Omega_m*(1.+z_pres)**3. + Omega_L)     #Hubble constant at z_pres (s^-1)
            tau_GP = (3.*n_bz*x_HI*(lambda_lya**3.)*gamma) / (2.*H_z)                   #Gunn-Peterson optical depth of all cells
            xi = ((1e-7*tau_GP)**(1./3.)) * (T_K**(-2./3.))                 #Xi parameter used in S_alpha calculation (all cells)
            log_Tk = np.nan_to_num(np.log10(T_K), neginf=-300.)             #Taking the log of T_k(z) for all scattering rate calculations
            kappa_HH = 3.1e-11*(T_K**(0.357)) * np.exp(-32./T_K)            #Scattering rate between H atoms
            n_e = (x_ez*n_Hz)/(1.-x_ez)                                                     #Electron (& therefore proton) number density (cm^-3)
            indicies_eH_lo, indicies_eH_hi = np.where(T_K<=1e4), np.where(T_K>1e4)          #Differentiate indicies where T_k > 1e4 K and <
            kappa_eH = np.zeros((N_side,N_side,N_side))                                     #Initialize e-H rate array 
            kappa_eH[indicies_eH_lo] = 10.**(-9.607 + 0.5*log_Tk[indicies_eH_lo] * np.exp(-((log_Tk[indicies_eH_lo])**4.5)/1800.))   #Scattering rate for electrons & H atoms (low T)
            kappa_eH[indicies_eH_hi] = 10.**(-9.607 + 0.5*4. * np.exp(-(4.**4.5)/1800.))    #And e-H rate for high T, below is protons & H atoms (my polyfit - Furlanetto & Furlanetto 2007b)
            kappa_pH = 10.**((-0.01833743*np.log10(T_K)**4.) + (0.15491927*np.log10(T_K)**3.) + (-0.31752573*np.log10(T_K)**2.) + (0.1205943*np.log10(T_K)) - 9.3771855)
            x_c = (T_star/(A_10*T_gamma)) * (kappa_HH*n_Hz + (kappa_eH+kappa_pH)*n_e)       #Collisional copuling parameter of all cells
            T_k_params = np.array([T_gamma, lambda_lya, T_star, gamma, A_10])               #Various values needed in T_S calculation
            T_S_all[:,:,:,iz] = Get_T_S(N_side, z_pres, T_K, J_a, xi, x_c, T_k_params)      #Calculate T_S for all cells
            T_S_avg[iz], T_S_sd[iz] = np.mean(T_S_all[:,:,:,iz]), np.std(T_S_all[:,:,:,iz])
            print(z_pres, np.min(T_S_all[:,:,:,iz]), np.mean(T_S_all[:,:,:,iz]), np.max(T_S_all[:,:,:,iz]))
        if overwrite == 1:
            np.save(save_path + '21cm/T_Spin_all' + special + '.npy', T_S_all)
            np.save(save_path + '21cm/T_S_Avg_SD' + special + '.npy', np.column_stack((T_S_avg, T_S_sd)))

    if get[3] == 1:
        print('Calculating differential brightness temperature')
        T_S_all = np.load(save_path + '21cm/T_Spin_all' + special + '.npy') #Spin temperature evolution of all cells
        T_b_all = np.zeros((N_side,N_side,N_side,len(z)))                   #Initialize 21-cm brightness temperature array
        T_b_avg, T_b_sd = np.zeros(len(z)), np.zeros(len(z))                #Also initialize mean/st. dev. of T_b(z) arrays
        tau_21_avg, tau_21_sd = np.zeros(len(z)), np.zeros(len(z))
        for iz in range(len(z)-1, -1, -1):                                  #Loop through time to calculate T_b(z) for all cells
            z_pres = round(z[iz],2)                                         #Current redshift
            n_bz = n_b0 * (1.+z_pres)**3. * (1.+all_delta*(Dz[iz]/Dz[0]))   #Redshifted baryon density
            T_gamma, T_spin = 2.726*(1.+z_pres), T_S_all[:,:,:,iz]                                      #Current CMB temperature & spin temp. of all cells
            H_z = 2.333e-18 * (h/0.72) * np.sqrt(Omega_m*(1.+z_pres)**3. + Omega_L)                     #Hubble constant at z_pres (s^-1)
            tau_21 = (3./(32.*np.pi)) * ((h_p*c**3.*A_10)/(k_B*T_spin*nu_21**2.)) * ((x_HI*n_bz)/H_z)   #21-cm optical depth for all cells
            tau_21_avg[iz], tau_21_sd[iz] = np.mean(tau_21), np.std(tau_21)
            T_b_all[:,:,:,iz] = ((T_spin-T_gamma)/(1.+z_pres)) * (1.-np.exp(-tau_21))           #Finally, calculate all T_b(z)
            T_b_avg[iz], T_b_sd[iz] = np.mean(T_b_all[:,:,:,iz]), np.std(T_b_all[:,:,:,iz])     #And its average/st. dev.
            print(z_pres, np.mean(T_spin), np.mean(tau_21))
            print(np.min(T_b_all[:,:,:,iz]), np.mean(T_b_all[:,:,:,iz]), np.max(T_b_all[:,:,:,iz]))
        if overwrite == 1:
            np.save(save_path + '21cm/Tau_21' + special + '.npy', np.column_stack((tau_21_avg, tau_21_sd)))
            np.save(save_path + '21cm/T_b_all' + special + '.npy', T_b_all)
            np.save(save_path + '21cm/T_b_Avg_SD' + special + '.npy', np.column_stack((T_b_avg, T_b_sd)))

    if get[4] == 1:
        print('Getting power spectra at all z')
        T_b_all, N_bins = np.load(save_path + '21cm/T_b_all' + special + '.npy') * 1e3, 50    #Load in T_b, convert to mK & define # of bins for PS
        Pk_all, k_all, nk_all = np.zeros((len(z),N_bins)), np.zeros((len(z),N_bins)), np.zeros((len(z),N_bins))     #Initialize arrays of all power spectra values
        for iz in range(len(z)-z_0, -1, -1):                                    #Loop through time to calculate all power spectra 
            z_pres, T_b = round(z[iz],2), T_b_all[:,:,:,iz]                     #Current redshift & T_b of all cells 
            [Pk_all[iz], k_all[iz], nk_all[iz]] = Power(192., T_b, T_b, N_bins) #Calculate power spectrum of T_b(z) 
            print(z_pres, Pk_all[iz])
        if overwrite == 1:
            np.save(save_path + '21cm/All_Power' + special + '.npy', Pk_all)
            np.save(save_path + '21cm/All_k' + special + '.npy', k_all)
            np.save(save_path + '21cm/All_N_k' + special + '.npy', nk_all)

    if get[5] == 1:
        print('Getting Lightcone')
        dx, dz = 3., 0.05                           #Length of subgrid cells [Mpc], change in redshift of z array
        T_b_all = np.load(save_path + '21cm/T_b_all' + special + '.npy') * 1e3  #4D array of 21-cm brightness temp results
        z_dists = np.load('./All_z_Distances.npy', allow_pickle=True)[0]        #Comoving distances from z=15 to all z'
        N_slice = int(round(z_dists[-1]/3.))        #Number of slices to be plotted in light cone (comoving dist from z=15-60 / subgrid cell box length)
        LC = np.zeros((N_side, N_side, N_slice))    #Initialize light cone -- N_cells x N_cells x N_slices from z=15-60
        z_LC = np.zeros(N_slice)                    #Also intialize array of z's used in lightcone
        for i in range(0, N_slice):                 #Loop through slices back in time to populate LC array
            slice_i = i % 64                        #Determine which slice of T_b cube we want to plot (iterative)
            all_dz = np.abs(z_dists-3.*i)           #Comoving distance = cell size * slice number
            iz_a = np.argmin(all_dz)                    #Find index of closest distance to this one in overall array
            dz_b = min(all_dz[iz_a-1], all_dz[iz_a+1])  #Determine which side of iz_a the distance fell (smaller of the two surrounding values)
            iz_b = np.argmin(np.abs(all_dz-dz_b))           #Use that distance value to find the index of the 2nd closest bin
            iz_0, iz_1 = min(iz_a, iz_b), max(iz_a, iz_b)   #Order them so that the smaller is iz_0
            z_fine = np.linspace(z[iz_0], z[iz_1], 100)     #Create finely spaced redshift array between these two
            H = 2.333e-18 * (h/0.72) * np.sqrt(Omega_m*(1.+z_fine)**3. + Omega_L)       #H(z), converted to 1/s
            dists_z = scipy.integrate.cumulative_trapezoid(c/H, z_fine) / (3.0857e24)   #Distance from z[iz_0] to all z_fine
            dists_z = np.concatenate(([0.], dists_z))       #Append dist from z[iz_0] to z[iz_0] = 0 to array of distances
            dists_z += z_dists[iz_0]                        #Then add distance from z=15 to z[iz_0] to those, making dists_z into distances from z=15
            z_i = z_fine[np.argmin(np.abs(dists_z-3.*i))]   #Find index of closest dist to actual comoving dist of this slice, get corresponding z
            iz_a = np.argmin(np.abs(z-z_i))                             #Now find closest index to z_i in overall z array
            dz_b = min(np.abs(z-z_i)[iz_a-1], np.abs(z-z_i)[iz_a+1])    #And determine which side of z[iz_a] z_i is found
            iz_b = np.argmin(np.abs(np.abs(z-z_i)-dz_b))                #Get its corresponding redshift
            iz_0, iz_1 = min(iz_a, iz_b), max(iz_a, iz_b)                       #Once again order them so that the smaller is iz_0
            T_b_0, T_b_1 = T_b_all[:,:,slice_i,iz_0], T_b_all[:,:,slice_i,iz_1] #Get the slices of T_b from both z[iz_0] & z[iz_1]
            dz_0, dz_1 = np.abs(z_i-z[iz_0]), np.abs(z_i-z[iz_1])               #Determine delta_z from each bin surrounding z_i to z_i
            LC[:,:,i] = (np.abs(dz-dz_0)*T_b_0 + np.abs(dz-dz_1)*T_b_1)/dz      #Get weighted sum of the two T_b slices & record in LC array
            z_LC[i] = z_i
        if overwrite == 1:
            np.save(save_path + '21cm/Lightcone.npy', LC)
            np.save(save_path + '21cm/Lightcone_z.npy', z_LC)

import numpy as np
import scipy.integrate, os
from scipy.optimize import curve_fit
cimport numpy as np
cimport cython

# This function is for using a spherical top-hat to smooth over the SFR cubes using FFTs --------------------------------------------
def smooth_cube(cube, radius):          #Function to take a 3D np.array & radius to get a spherical top-hat average using convolution
    cdef int cube_size
    cdef list dists, corners
    cdef np.ndarray[np.int_t,ndim=1] work_shape
    cdef np.ndarray[np.complex128_t,ndim=3] cube_k, kernel_k, result0
    cdef np.ndarray[np.float64_t,ndim=3] work_cube, dist, kernel, result

    work_cube = cube.copy()             #Add padding around edges of cube to remove periodic B.C.s
    cube_k = np.fft.fftn(work_cube)     #And take the FFT
    cube_size = len(work_cube[:,0,0])   #Save the cube size
    dists = []
    xg, yg, zg = np.meshgrid(np.arange(cube_size), np.arange(cube_size), np.arange(cube_size))  #Get distances from center cube & corner cell IDs
    corners = [[0,0,0],[0,0,cube_size],[0,cube_size,0],[0,cube_size,cube_size],[cube_size,0,0],[cube_size,0,cube_size],[cube_size,cube_size,0],[cube_size,cube_size,cube_size]]
    for i in range(len(corners)):               #Get distances from center to each corner cell
        dists.append(np.sqrt((corners[i][0] - xg)**2. + (corners[i][1] - yg)**2. + (corners[i][2] - zg)**2.))
    dist = np.amin(dists, axis=0)               #Next make a cube containing the distances from each cell to the closest corner
    
    work_shape = np.array((work_cube.shape[0], work_cube.shape[1], work_cube.shape[2]))  #Get lengths of sides to initialize kernel
    kernel = np.zeros(work_shape)               #Initialize the kernel as an equal size cube of zeros...
    kernel[(dist < radius)] = 1                 #Set any point within radius from the center to be 1
    kernel = kernel / np.sum(kernel)            #Divide the kernel by the number of points to get an average
    kernel_k = np.fft.fftn(kernel)              #FFT the kernel cube
    result0 = np.fft.ifftn(cube_k * kernel_k)   #Now that we have the kernel, perform the convolution
    result = np.real(result0)                   #Remove imaginary components
    result[result < 1e-15] = 0.0                #And any negative values/numerical errors
    del work_cube, cube_k, kernel, kernel_k, result0
    return(result)

# This function is for smoothing the SFR values over the lifetime of the star -------------------------------------------------------
def Smooth_SFRs(int N, int z_iz, np.ndarray[np.float64_t,ndim=1] z, np.ndarray[np.float64_t,ndim=1] t, np.ndarray[np.float64_t,ndim=4] SFR_III_all, np.ndarray[np.float64_t,ndim=4] SFR_II_all, int iz_stop): 
    cdef int iz, i, t_iz, N_steps
    cdef double t_3
    cdef np.ndarray[np.float64_t,ndim=3] SFR_II_z, SFR_III_z, SFR_II_add, SFR_III_add
    cdef np.ndarray[np.float64_t,ndim=4] SFR_II_smooth, SFR_III_smooth

    SFR_II_smooth = np.zeros((N,N,N,len(z)))
    SFR_III_smooth = np.zeros((N,N,N,len(z)))           #Initalize smoothed SFR arrays up to current z step
    for iz in range(iz_stop, z_iz, -1):                 #Then loop through redshift steps so far
        SFR_II_z = SFR_II_all[:,:,:,iz]                 #Isolate SFR(z) for both stellar populations
        SFR_III_z = SFR_III_all[:,:,:,iz]
        t_3 = t[iz] + 3e6                               #Current Hubble time + 3 Myr stellar lifetime
        t_iz = np.argmin(np.abs(t-t_3))                 #Find the corresponding index
        N_steps = iz - t_iz                             #And the number of z steps which that timespan covers
        SFR_II_add = SFR_II_z/N_steps                   #Get SFR(z)/N_steps for both populations...
        SFR_III_add = SFR_III_z/N_steps                 #So that we can add these values to each of the following N_steps
        for i in range(0, N_steps):                     #Then loop through those steps to add the SFRs to...
            SFR_II_smooth[:,:,:,iz-i] += SFR_II_add     #Add SFR_i_add to indices from z[iz] to z[iz+N_steps]
            SFR_III_smooth[:,:,:,iz-i] += SFR_III_add   #To both stellar populations to smooth SFR(z)
    return(SFR_III_smooth, SFR_II_smooth)

# This function is for interpolating the SFRD values between smoothed shells of SFR for Get_J ---------------------------------------
def Interp_SFRs(int N, np.ndarray[np.float64_t,ndim=4] SFR_II, np.ndarray[np.float64_t,ndim=4] SFR_III, np.ndarray[np.float64_t,ndim=1] LW_dists, np.ndarray[np.float64_t,ndim=1] z_dists, np.ndarray[np.float64_t,ndim=1] z_higher):
    cdef int iz, ii, ij, ik
    cdef np.ndarray[np.float64_t,ndim=4] cube_II, cube_III

    cube_II, cube_III = np.zeros((N,N,N,len(z_higher))), np.zeros((N,N,N,len(z_higher)))    #Initialize 4D cubes of smoothed SFR values
    for ii in range(0, N):
      for ij in range(0, N):        #Loop through every cell and interpolate
        for ik in range(0, N):
            SFR_cell_II, SFR_cell_III = SFR_II[ii,ij,ik], SFR_III[ii,ij,ik]     #This cell's smoothed SFR(z') arrays
            cube_II[ii,ij,ik] = np.interp(z_higher, z_dists, SFR_cell_II)       #Interpolate SFR values this cell
            cube_III[ii,ij,ik] = np.interp(z_higher, z_dists, SFR_cell_III)     #For both stellar populations
    cube_II /= 27.
    cube_III /= 27.                 #Divide cubes by cell volume in Mpc^3 to get SFRDs
    cube_III[cube_III<1e-15] = 0.0  #Don't let PopIII SFRD be below zero
    return cube_II, cube_III        #Then return both SFRD cubes to Get_J

# This function is for calculating the J_LW background intensity seen by all cells at each z step -----------------------------------
def Get_J(double z_pres, np.ndarray[np.float64_t,ndim=1] z, np.ndarray[np.float64_t,ndim=1] t, np.ndarray[np.float64_t,ndim=1] f_LW, int N, np.ndarray[np.float64_t,ndim=1] LW_vals, np.ndarray[np.float64_t,ndim=4] SFR_III_0, np.ndarray[np.float64_t,ndim=4] SFR_II_0, int z_iz, str save_path, np.ndarray[np.float64_t,ndim=1] LW_dists, np.ndarray[np.float64_t,ndim=1] z_dists, int iz_stop):
    cdef int iz, ii, ij, ik, first_i, final_i, N_iz, N_LW
    cdef double H_0_s, Omega_m, m_proton, eta_II, eta_III, E_LW, dnu_LW, c, N_cells, V_0, V_1
    cdef np.ndarray[np.float64_t,ndim=1] dt_dz, Const, dt
    cdef np.ndarray[np.float64_t,ndim=3] SFR_II, SFR_III, smooth_II, smooth_III, J_LW
    cdef np.ndarray[np.float64_t,ndim=4] SFR_III_all, SFR_II_all, SFR_cube_II, SFR_cube_III, SFRD_cube_II, SFRD_cube_III, SFRD_II, SFRD_III, epsilon_II, epsilon_III, epsilon, Js, sum_add, testing

    H_0_s, Omega_m, m_proton, eta_II, eta_III, E_LW, dnu_LW, c = LW_vals[0], LW_vals[1], LW_vals[2], LW_vals[3], LW_vals[4], LW_vals[5], LW_vals[6], LW_vals[7]    #Constants used in LW calculation
    N_iz, N_LW = min(len(f_LW),len(z)-z_iz), len(z_dists)                           #Number of previous cubes to smooth over & of previous z steps to get LW for & constant N_smoothing shells
    z_higher = z[z_iz:z_iz+N_iz]                                                    #Higher redshifts
    SFRD_cube_II, SFRD_cube_III = np.zeros((N,N,N,N_iz)), np.zeros((N,N,N,N_iz))    #Initialize cubes of SFRD values over time (NxNxN cube where each index is array for SFRD(z'))
    SFR_cube_II, SFR_cube_III = np.zeros((N,N,N,N_LW)), np.zeros((N,N,N,N_LW))          #Initialize cubes of smoothed SFR cube histories -- used in interpolation to populate SFRD cubes
    SFR_III_all, SFR_II_all = Smooth_SFRs(N, z_iz, z, t, SFR_III_0, SFR_II_0, iz_stop)  #Smooth SFRs over PopIII stellar lifetime (3 Myr)
    for iz in range(0, N_LW):                                                           #Loop through constant distance shells and smooth 
        N_cells = LW_dists[iz]/3.                               #Number of cells over which to smooth SFR cubes
        iz_smooth = np.argmin(np.abs(z-z_dists[iz]))            #Index of corresponding redshift to smoothing distance
        SFR_II = SFR_II_all[:,:,:,iz_smooth]                    #Isolate raw SFR values for this z step
        SFR_III = SFR_III_all[:,:,:,iz_smooth]                  #For both stellar populations
        SFR_cube_II[:,:,:,iz] = smooth_cube(SFR_II, N_cells)    #Populate SFR cubes with smoothed SFR for this constant z distance
        SFR_cube_III[:,:,:,iz] = smooth_cube(SFR_III, N_cells)  #For both stellar populations
    SFRD_cube_II, SFRD_cube_III = Interp_SFRs(N, SFR_cube_II, SFR_cube_III, LW_dists, z_dists, z_higher[:N_iz]) 
    SFRD_cube_III[SFRD_cube_III < 1e-15] = 0.0                              #Make sure that smoothing didn't cause SFRD_III to be negative anywhere
    SFRD_II, SFRD_III = 2.146e-48*SFRD_cube_II, 2.146e-48*SFRD_cube_III     #Converting from M_sol/yr/Mpc^3 to g/s/cm^3
    epsilon_II = SFRD_II * (1./m_proton) * eta_II * E_LW / dnu_LW
    epsilon_III = SFRD_III * (1./m_proton) * eta_III * E_LW / dnu_LW        #Get LW emissivity contributions from both stellar populations
    epsilon = np.add(epsilon_II, epsilon_III)                               #Sum contributions from both stellar populations
    del SFRD_cube_II, SFRD_cube_III, epsilon_II, epsilon_III                #Delete large arrays for efficiency
    dt_dz = 1./(H_0_s*(1.+z_higher[:N_iz])*(np.sqrt(Omega_m*((1.+z_higher[:N_iz])**3.))))   #|dt_H/dz'| term
    Const = ((c*((1.+z_pres)**3.)) / (4.*np.pi)) * 0.05 * dt_dz             #Constant outside of integral * z step size * |dt_H/dz'| term
    Js = Const*f_LW[:N_iz]*epsilon                                          #Array of contributions to J_LW(distance)
    J_LW = np.sum(Js, axis=3) / 1e-21    #Sum contributions along fourth axis & get in units of 10^-21 erg/s/cm^2/Hz/Sr
    J_LW[J_LW < 0.0] = 0.0
    J_LW[J_LW < 1e-6] += 1e-6
    return(J_LW)

# This function is for calculating M_crit using Mihir's model (Kulkarni et al. 2021) ------------------------------------------------
def Get_M_Crit(np.ndarray[np.float64_t,ndim=3] J, np.ndarray[np.float64_t,ndim=3] v_bc, double z, np.ndarray[np.float64_t,ndim=1] J0_M_crit, np.ndarray[np.int_t,ndim=3] all_vbc_i, double M_a):      #This function is for Mihir's fit for critical mass
    cdef np.ndarray[np.float64_t,ndim=3] alpha, M_20, M_crit_0, J_0_cube, M_crit_1

    alpha = 1.64 * ((1.0 + J)**0.36) * ((1.0 + v_bc/30.)**-0.62) * ((1.0 + (J*v_bc)/3.)**0.13)
    M_20 = 1.96*1e5 * ((1.0 + J)**0.8) * ((1.0 + v_bc/30.)**1.83) * ((1.0 + (J*v_bc)/3.)**-0.06)
    M_crit_0 = M_20 * ((1.0+z)/21.0)**-alpha
    J_0_cube = J0_M_crit[all_vbc_i]             #3D array of J_0 values by vbc
    M_crit_1 = np.maximum(M_crit_0, J_0_cube)   #Replace those that are too low
    M_crit_1[M_crit_1 > M_a] = M_a              #Use M_a if M_crit_H2 > M_a
    del alpha, M_20, M_crit_0, J_0_cube
    return(M_crit_1)

# Above are the NN models and codes to get J_LW(z), M_crit, & smooth data cubes over various distances ---- Below is the main self-consistent model ----------------

def Simulation(double f_m, double fIII, double fII, str special=''):
    cdef int iz, ii, ij, ik, N_iz, delta_i, l_bound_III, u_bound_III, u_bound_II_a, end_a, end_i, end_m
    cdef str box_path, save_path
    cdef double t_delay, V_com, H_0, Omega_m, Omega_b, c, H_0_s, eta_III, eta_II, m_proton, h_planck, E_LW_l_eV, E_LW_u_eV, E_LW_l, E_LW_u, E_LW, nu_l, nu_u, dnu_LW, A_p, a_p, q_p, dm, G, Omega_l, rho_c, rho_0, rho_b, F_const, z_pres, M_a, M_i, dF_dt_a, dF_dt_i, dF_dt_m
    cdef np.ndarray[np.int_t,ndim=3] all_vbc_i, all_delta_i 
    cdef np.ndarray[np.float64_t,ndim=1] z, t, t_recent, M_a_z, M_i_z, deltas, vbcs, masses, LW_vals, M_crit_0, DNDM, bounds, bounds_III, bounds_II_a, bounds_II_i, DNDM_III, DNDM_II_a, DNDM_II_i, coeffs_a, coeffs_i, coeffs_m, F_coll_a_0, F_coll_i_0, F_coll_m_0
    cdef np.ndarray[np.float64_t,ndim=2] J_0, DNDM_all
    cdef np.ndarray[np.float64_t,ndim=3] all_delta, all_vbc
    cdef np.ndarray[np.float64_t,ndim=4] SFR_III, SFR_II, J_z, M_crit_z, F_coll_a_z, F_coll_i_z, F_coll_m_z

    N_side = 64
    z = np.linspace(15., 60., 901)          #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))      #Hubble time array
    M_a_z = 5.4e7*(((1.0+z)/11.0)**-1.5)    #Atomic cooling masses
    M_i_z = 1.5e8*(((1.0+z)/11.0)**-1.5)    #Ionised IGM feedback mass
    deltas = np.load('./Delta_Bins_64_400.npy')    #Array of binned overdensity values
    vbcs = 30.* np.linspace(0.0, 3.0, 301)  #Streaming velocities
    masses = np.linspace(5.6, 9.5, 40)      #Mass bins array
    t_delay, V_com = 1e7, 3.0**3.0          #Delay time for PopII SF & cell volume
    box_path = './21cm/Box_' + str(round(3.*N_side,1)) + '_' + str(N_side) + '_z15.0/'
    save_path = './Results/Integral_' + str(N_side) + special + '/'
    try:
        os.mkdir(save_path)
    except OSError:
        print(OSError)

    coeffs = np.load('./Schaerer_PopIII_Fit_Coeffs.npy', allow_pickle=True) #Fitting coefficients for the PopIII stellar lifetime...
    coeffs_life, coeffs_eLW = coeffs[0], coeffs[3]  #And # of ion/LW photons per baryon 
    fit_life, fit_eLW = 0., 0.                      #Initialize fit value for each property
    for i in range(0, len(coeffs_life)):
        fit_life += coeffs_life[i]*np.log10(fIII)**float(len(coeffs_life)-i-1)   #Calculate fit values for each property
    for i in range(0, len(coeffs_eLW)):
        fit_eLW += coeffs_eLW[i]*np.log10(fIII)**float(len(coeffs_eLW)-i-1)
    print('(Log) PopIII Lifetime, eta_LW = ', fit_life, fit_eLW)
    fit_life, fit_eLW = 10.**(fit_life), 10.**(fit_eLW)              #Raise fit to 10^fit 

    H_0, Omega_m, Omega_b, c = 67., 0.32, 0.049, 29979245800.   #Hubble constant, matter density param, speed of light (cm/s)
    eta_II, eta_III, H_0_s = 9690., fit_eLW, H_0/3.0857e19      #LW/ionizing photons per baryon by stellar pop & Hubble const in 1/s
    m_proton, h_planck = 1.673e-24, 6.626e-27                   #Mass of a proton in g & Planck's Constant in cgs
    E_LW_l_eV, E_LW_u_eV = 11.2, 13.6                                   #LW energy limits in eV
    E_LW_l, E_LW_u = E_LW_l_eV * 1.6022e-12, E_LW_u_eV * 1.6022e-12     #LW energy limits in erg
    E_LW = (E_LW_u+E_LW_l)/2.                                           #Average energy of LW in erg
    nu_l, nu_u = E_LW_l/h_planck, E_LW_u/h_planck                       #Frequencies of LW energy limits
    dnu_LW = nu_u - nu_l                                                             #Difference of frequency limits
    LW_vals = np.array([H_0_s, Omega_m, m_proton, eta_II, eta_III, E_LW, dnu_LW, c]) #Values needed to run Get_J
    A_p, a_p, q_p, dm, G, H_0 = 0.322, 0.75, 0.3, 100., 4.494e-33, 6.847e-11         #Parameters for getting dn/dM (G in Mpc^3*Msun^-1*yr^-2, & H_0 in yr^-1)
    Omega_l, rho_c = 1.- Omega_m, (3.*H_0**2.)/(8.*np.pi*G)                     #DE fraction & Critical density
    rho_0, rho_b, F_const = rho_c*Omega_m, rho_c*Omega_b, 1./(Omega_m*rho_c)    #Density of all matter, baryons & Constant outside fraction Integral

    J_0 = np.load('./J_LW_0/J_0/All_J_0.npy')               #J_LW = 0 M_crit values by sigma_vbc
    all_delta = np.load('./Overdensity_Field_64_192.npy')   #All cell overdensities
    all_vbc = np.abs(np.load('./Vbc_Field_64_192.npy'))     #All cell streaming velocities
    all_delta_i = np.load('./delta_i.npy').astype(int)      #All cell overdensity bins
    all_vbc_i = np.load('./vbc_i.npy').astype(int)          #And cell vbc bins
    all_f_LW = np.load('./All_f_LW.npy', allow_pickle=True)             #Attenuation arrays for each z
    dists_LW = np.load('./LW_Smoothing_Scales.npy', allow_pickle=True)  #Distances over which SFR cube is smoothed at each z step
    dists_z = np.load('./LW_Smoothing_zs.npy', allow_pickle=True)       #Redshifts at those distances
    M_crit_0 = np.load('./J_LW_0/J_0/J_0_0.0.npy')                      #M_crit(z) for vbc=J=0 (lower limit of PopIII bounds)

    SFR_III = np.zeros((N_side, N_side, N_side, len(z)))    #Initialize arrays that are the size of our 3D cube with each element...
    SFR_II = np.zeros((N_side, N_side, N_side, len(z)))     #Being a z-length array of PopII/III SFRs to populate as we step through time...
    J_z = np.zeros((N_side, N_side, N_side, len(z)))        #Initialize these for both stellar populations, the LW background, and M_crit...
    M_crit_z = np.zeros((N_side, N_side, N_side, len(z)))   #Cube of critical masses over time
    F_coll_a_z = np.zeros((N_side, N_side, N_side, len(z)))
    F_coll_i_z = np.zeros((N_side, N_side, N_side, len(z))) #Collapse fraction for SFRD calculation
    F_coll_m_z = np.zeros((N_side, N_side, N_side, len(z)))

    def poly_func(z,A,B,C):
        return A*((1.+z)**2.) + B*(1.+z) + C    #Define polynomial function to curve-fit F_collapse

    for iz in range(len(z)-3, -1, -1):          #Begin stepping through time to simulate 
        z_pres = round(z[iz], 2)                #Present redshift
        M_a, M_i = M_a_z[iz], M_i_z[iz]         #Atomic cooling & Ionization masses
        dt = t[iz] - t[iz+1]                    #Change in time since last step
        iz_stop = min(898, np.argmin(np.abs(z-dists_z[iz][-1]))+5)  #Redshift index at which Smooth_SFRs should stop (plus some buffer for 3 Myr smoothing)
        J_z[:,:,:,iz] = Get_J(z_pres, z, t, all_f_LW[iz], N_side, LW_vals, np.copy(SFR_III), SFR_II, iz, save_path, dists_LW, np.array(dists_z[iz]), iz_stop)   #LW Background calculation for this z step
        M_crit_z[:,:,:,iz] = Get_M_Crit(J_z[:,:,:,iz], all_vbc, z_pres, J_0[:,iz], all_vbc_i, M_a)   #M_crit = lower bound for popIII integration
        print(z_pres, np.mean(J_z[:,:,:,iz]), np.mean(M_crit_z[:,:,:,iz]))
        DNDM_all = np.load('./Int_DNDMs/Num_Halos_z_' + str(z_pres) + '.npy')           #dn/dM for this redshift, from M_crit_0(z) to 1e9.1
        for ii in range(0, N_side):
            for ij in range(0, N_side):                     #Loop through all cells in box
                for ik in range(0, N_side):
                    delta, delta_i = all_delta[ii,ij,ik], all_delta_i[ii,ij,ik]         #This cell's overdensity & index
                    DNDM = DNDM_all[:,delta_i]                                          #Isolate relevant dn/dM array from overall list
                    bounds = np.logspace(np.log10(M_crit_0[iz]), 13., 750)              #And get the M_halo range used in that dn/dM
                    l_bound_III = np.argmin(np.abs(bounds-M_crit_z[ii,ij,ik,iz]))       #Find the index of M_crit(z) in M_crit_0
                    u_bound_III = np.argmin(np.abs(bounds-f_m*M_crit_z[ii,ij,ik,iz]))   #Find the index of f_m*M_crit(z)
                    u_bound_II_a = np.argmin(np.abs(bounds-M_i))                        #And the index of M_i(z)
                    bounds_III = bounds[l_bound_III:u_bound_III]
                    bounds_II_a = bounds[u_bound_III:u_bound_II_a]  #Then use bounds to get mass regimes for each stellar population
                    bounds_II_i = bounds[u_bound_II_a:]
                    DNDM_III = DNDM[l_bound_III:u_bound_III]
                    DNDM_II_a = DNDM[u_bound_III:u_bound_II_a]      #And the relevant dn/dM values
                    DNDM_II_i = DNDM[u_bound_II_a:]
                    F_coll_a_z[ii,ij,ik,iz] = F_const * np.trapz(bounds_II_a*DNDM_II_a, bounds_II_a) #Then calculate F_collapses
                    F_coll_i_z[ii,ij,ik,iz] = F_const * np.trapz(bounds_II_i*DNDM_II_i, bounds_II_i)
                    F_coll_m_z[ii,ij,ik,iz] = np.trapz(DNDM_III, bounds_III)
                    if iz < 700:                                                        #At lower redshifts, fit the F_collapse values to avoid unphysical feedback
                      end_a, end_i, end_m = np.max(np.nonzero(F_coll_a_z[ii,ij,ik])[0]), np.max(np.nonzero(F_coll_i_z[ii,ij,ik])[0]), np.max(np.nonzero(F_coll_m_z[ii,ij,ik])[0])
                      coeffs_a, covar = curve_fit(f=poly_func, xdata=z[iz:end_a], ydata=np.log10(F_coll_a_z[ii,ij,ik,iz:end_a]))
                      coeffs_i, covar = curve_fit(f=poly_func, xdata=z[iz:end_i], ydata=np.log10(F_coll_i_z[ii,ij,ik,iz:end_i]))    #Fit each F_collapse range
                      coeffs_m, covar = curve_fit(f=poly_func, xdata=z[iz:end_m], ydata=np.log10(F_coll_m_z[ii,ij,ik,iz:end_m]))
                      F_coll_a_0 = 10.**(coeffs_a[0]*((1.+z)**2.) + coeffs_a[1]*(1.+z) + coeffs_a[2])
                      F_coll_i_0 = 10.**(coeffs_i[0]*((1.+z)**2.) + coeffs_i[1]*(1.+z) + coeffs_i[2])   #Calculate F_collapse using fits
                      F_coll_m_0 = 10.**(coeffs_m[0]*((1.+z)**2.) + coeffs_m[1]*(1.+z) + coeffs_m[2])
                      dF_dt_a = max((F_coll_a_0[iz] - F_coll_a_0[iz+1]) / dt, 0.)
                      dF_dt_i = max((F_coll_i_0[iz] - F_coll_i_0[iz+1]) / dt, 0.)   #Determine dF_coll/dt to be change in fit values/dt (or zero if negative)
                      dF_dt_m = max((F_coll_m_0[iz] - F_coll_m_0[iz+1]) / dt, 0.)
                    else:
                      dF_dt_a = max((F_coll_a_z[ii,ij,ik,iz] - F_coll_a_z[ii,ij,ik,iz+1]) / dt, (F_coll_a_z[ii,ij,ik,iz+1] - F_coll_a_z[ii,ij,ik,iz+2]) / (t[iz+1] - t[iz+2]), 0.)  #Take dF_coll/dt to be max of 
                      dF_dt_i = max((F_coll_i_z[ii,ij,ik,iz] - F_coll_i_z[ii,ij,ik,iz+1]) / dt,	(F_coll_i_z[ii,ij,ik,iz+1] - F_coll_i_z[ii,ij,ik,iz+2]) / (t[iz+1] - t[iz+2]), 0.)  #current value, previous value,
                      dF_dt_m = max((F_coll_m_z[ii,ij,ik,iz] - F_coll_m_z[ii,ij,ik,iz+1]) / dt,	(F_coll_m_z[ii,ij,ik,iz+1] - F_coll_m_z[ii,ij,ik,iz+2]) / (t[iz+1] - t[iz+2]), 0.)  #and zero to avoid negatives
                    SFR_II[ii,ij,ik,iz] = rho_b * fII * (dF_dt_a+dF_dt_i) * V_com   #Calculate PopII SFR from dF_coll/dt
                    SFR_III[ii,ij,ik,iz] = fIII * dF_dt_m * V_com                   #And PopIII SFR

        if iz % 100 == 0:
            np.save(save_path + '/J_z_all.npy', J_z)
            np.save(save_path + '/M_crit_all.npy', M_crit_z)
            np.save(save_path + '/SFR_II_all.npy', SFR_II)
            np.save(save_path + '/SFR_III_all.npy', SFR_III)
            np.save(save_path + '/F_coll_a.npy', F_coll_a_z)
            np.save(save_path + '/F_coll_i.npy', F_coll_i_z)
            np.save(save_path + '/F_coll_m.npy', F_coll_m_z)
        print(z_pres, np.mean(J_z[:,:,:,iz]), np.mean(M_crit_z[:,:,:,iz]))
        print(np.mean(SFR_III[:,:,:,iz]/V_com), np.mean(SFR_II[:,:,:,iz]/V_com))
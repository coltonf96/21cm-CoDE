import numpy as np
import torch, os
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
cimport numpy as np
cimport cython

class SimpleNet2b(nn.Module):       #This is the Neural Net for BURSTY PopII ---------------------------------------------------------------
    def __init__(self):
        super(SimpleNet2b, self).__init__() #An apparently necessary line -- super("ClassName", self).__init__()
        self.fc1 = nn.Linear(2, 50)         #Two input channel to 50 variables
        self.fc2 = nn.Linear(50, 50)        #These 50 each talk to the next 50
        self.fc3 = nn.Linear(50, 1)         #And then gives one output
    def forward(self, x):                   #Forward pass
        x = F.relu(self.fc1(x))   #Passes x through self.fc1 in above function : N_inputs-50
        x = F.relu(self.fc2(x))   #Then through self.fc2 : 50-50
        x = self.fc3(x)           #Finally, back to one output : 50-1
        return x

class SimpleNet2s(nn.Module):       #This is the Neural Net for STEADY PopII ---------------------------------------------------------------
    def __init__(self):
        super(SimpleNet2s, self).__init__() #An apparently necessary line -- super("ClassName", self).__init__()
        self.fc1 = nn.Linear(3, 50)         #Three input channel to 50 variables
        self.fc2 = nn.Linear(50, 50)        #These 50 each talk to the next 50
        self.fc3 = nn.Linear(50, 1)         #And then gives one output
    def forward(self, x):                   #Forward pass
        x = F.relu(self.fc1(x))   #Passes x through self.fc1 in above function : N_inputs-50
        x = F.relu(self.fc2(x))   #Then through self.fc2 : 50-50
        x = self.fc3(x)           #Finally, back to one output : 50-1
        return x

class SimpleNet3(nn.Module):      #And for PopIII -----------------------------------------------------------------------------------------
    def __init__(self):
        super(SimpleNet3, self).__init__()  #An apparently necessary line -- super("ClassName", self).__init__()
        self.fc1 = nn.Linear(5, 50)         #Five input channel to 50 variables
        self.fc2 = nn.Linear(50, 50)        #These 50 each talk to the next 50
        self.fc3 = nn.Linear(50, 1)         #And then gives one output
    def forward(self, x):                   #Forward pass
        x = F.relu(self.fc1(x))   #Passes x through self.fc1 in above function : N_inputs-50
        x = F.relu(self.fc2(x))   #Then through self.fc2 : 50-50
        x = self.fc3(x)           #Finally, back to one output : 50-1
        return x 

device = ('cuda' if torch.cuda.is_available()
  else 'cpu')                     #Try to use a GPU if possible

# This function is for using a spherical top-hat to smooth over the SFR cubes using FFTs --------------------------------------------
def smooth_cube(cube, radius):
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
    kernel = kernel / np.sum(kernel)                    #Divide the kernel by the number of points to get an average
    kernel_k = np.fft.fftn(kernel)                      #FFT the kernel cube
    result0 = np.fft.ifftn(cube_k * kernel_k)           #Now that we have the kernel, perform the convolution
    result = np.real(result0)                           #Remove imaginary components
    result[result < 1e-15] = 0.0                        #And any negative values/numerical errors
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

    H_0_s, Omega_m, eta_II, eta_III, c, C_emis = LW_vals[0], LW_vals[1], LW_vals[2], LW_vals[3], LW_vals[4], LW_vals[5]    #Constants used in LW calculation
    z_higher, N_iz, N_LW = z[z_iz:], len(f_LW), len(z_dists)                        #All higher z values, # of previous z steps to get to LW horizon & const N_smoothing shells (fid=20)
    SFR_cube_II, SFR_cube_III = np.zeros((N,N,N,N_LW)), np.zeros((N,N,N,N_LW))      #Initialize cubes of smoothed SFR cube histories -- used in interpolation to populate SFRD cubes
    SFRD_cube_II, SFRD_cube_III = np.zeros((N,N,N,N_iz)), np.zeros((N,N,N,N_iz))    #Initialize cubes of SFRD values over time (NxNxN cube where each index is array for SFRD(z'))
    SFR_III_all, SFR_II_all = Smooth_SFRs(N, z_iz, z, t, SFR_III_0, SFR_II_0, iz_stop)  #Smooth SFRs over PopIII stellar lifetime (fiducial=3 Myr)
    for iz in range(0, N_LW):                                                       #Now loop through constant distance shells and smooth the cube over each
        N_cells = LW_dists[iz]/3.                                                   #Number of cells over which to smooth SFR cubes (divided by 3 Mpc cell size)
        iz_smooth = np.argmin(np.abs(z-z_dists[iz]))                                #Index of corresponding redshift at smoothing distance
        SFR_II = SFR_II_all[:,:,:,iz_smooth]                                        #Isolate raw SFR values for this z step
        SFR_III = SFR_III_all[:,:,:,iz_smooth]                                      #For both stellar populations
        SFR_cube_II[:,:,:,iz] = smooth_cube(SFR_II, N_cells)                        #Smooth the SFR cube on this constant z distance scale
        SFR_cube_III[:,:,:,iz] = smooth_cube(SFR_III, N_cells)                              #For both stellar populations
    SFRD_cube_II, SFRD_cube_III = Interp_SFRs(N,SFR_cube_II,SFR_cube_III,LW_dists,z_dists,z_higher[:N_iz])     #Interpolate SFRDs at intermediate z steps
    SFRD_II, SFRD_III = 2.146e-48*SFRD_cube_II, 2.146e-48*SFRD_cube_III                     #Convert from M_sol/yr/Mpc^3 to g/s/cm^3
    epsilon_II, epsilon_III = SFRD_II*C_emis*eta_II, SFRD_III*C_emis*eta_III                #Get LW emissivity contributions from both stellar populations
    epsilon = np.add(epsilon_II, epsilon_III)                                               #Sum contributions from both stellar populations
    del SFRD_cube_II, SFRD_cube_III, epsilon_II, epsilon_III                                #Delete large arrays for efficiency
    dt_dz = 1./(H_0_s*(1.+z_higher[:N_iz])*(np.sqrt(Omega_m*((1.+z_higher[:N_iz])**3.))))   #|dt_H/dz'| term
    Const = ((c*((1.+z_pres)**3.)) / (4.*np.pi)) * 0.05 * dt_dz                             #Constant outside of integral * z step size * |dt_H/dz'| term
    Js = Const*f_LW*epsilon                                         #Array of contributions to J_LW(distance)
    J_LW = np.sum(Js, axis=3) / 1e-21                               #Sum contributions along fourth axis & get in units of 10^-21 erg/s/cm^2/Hz/Sr
    J_LW[J_LW < 0.0] = 0.0                                          #Make sure there are no negative J_LW cells
    J_LW[J_LW < 1e-6] += 1e-6                                       #And add LW floor to those below it
    return(J_LW)

# This function is for calculating M_crit using Mihir's model (Kulkarni et al. 2021) ------------------------------------------------
def Get_M_Crit(np.ndarray[np.float64_t,ndim=3] J, np.ndarray[np.float64_t,ndim=3] v_bc, double z, np.ndarray[np.float64_t,ndim=1] J0_M_crit, np.ndarray[np.int_t,ndim=3] all_vbc_i, double M_a, double M_crit_mod):
    cdef np.ndarray[np.float64_t,ndim=3] alpha, M_20, M_crit_0, J_0_cube, M_crit_1

    alpha = 1.64 * ((1.+J)**0.36) * ((1.+v_bc/30.)**-0.62) * ((1.+(J*v_bc)/3.)**0.13)       #Power exponent on z-dependency
    M_20 = 1.96*1e5 * ((1.+J)**0.8) * ((1.+v_bc/30.)**1.83) * ((1.+(J*v_bc)/3.)**-0.06)     #Coefficient M_crit(z=20) term
    M_crit_0 = M_20 * ((1.+z)/21.)**-alpha      #Calculate M_Crit(z)
    M_crit_0 *= M_crit_mod                      #Multiply by modification factor (for Hazlett+25b comparison)
    J_0_cube = J0_M_crit[all_vbc_i]*M_crit_mod  #3D array of M_Crit(J_LW=0) values by vbc
    M_crit_1 = np.maximum(M_crit_0, J_0_cube)   #Replace values in cells that are too low
    M_crit_1[M_crit_1 > M_a] = M_a              #Use M_a in cells where M_crit_H2 > M_a
    del alpha, M_20, M_crit_0, J_0_cube         #Delete large cubes for efficiency
    return(M_crit_1)

# Above are the NN models and codes to get J_LW(z), M_crit, & smooth data cubes over various distances ---- Below is the main self-consistent model ----------------

def Simulation(str special, str NN_special, int N_star=4, double star_m0=35.69, float t_delay=1e7, float M_crit_mod=1.0):
    cdef int N_side, N_EPOCHS_III, N_EPOCHS_II, iz, delay_iz, N_iz, ii, ij, ik, emulation, delta_i, iz_stop, i
    cdef str base_path, save_path, emulating, suffix
    cdef double V_com, H_0, Omega_m, c, H_0_s, eta_II, eta_III, m_proton, h_planck, E_LW_l_eV, E_LW_u_eV, E_LW_l, E_LW_u, E_LW_eV, E_LW, nu_l, nu_u, dnu_LW, z_pres, dt, t_delay_ago, M_a, delta, int_M, int_J, log_M, log_J, log_J_LW, log_MMH, vbc, dII_dz, Mcrit_delay, log_N_t, log_M_t, fit_life, fit_ion, fit_LW, fit_eLW, fit_eta
    cdef np.ndarray[np.int_t,ndim=3] mergers, all_delta_i, all_vbc_i
    cdef np.ndarray[np.float64_t,ndim=1] z, t, vbcs, vbc_z, LW_vals, MMH, inputs_III_0, M_crit_cell, J_LW_cell, inputs_II_0, N_fid_III, N_fid_II, t_recent, deltas, deltas0, coeffs_life, coeffs_ion, coeffs_LW, coeffs_eLW, coeffs_eta
    cdef np.ndarray[np.float64_t,ndim=2] J_0, MMHs
    cdef np.ndarray[np.float64_t,ndim=3] t_III_all, J, M_crit, all_vbc, all_delta
    cdef np.ndarray[np.float64_t,ndim=4] SFR_III, SFR_II, Pop_III, Pop_II_B, Pop_II_S, J_z, M_crit_z, Pop_III_Smooth, Pop_II_Smooth

    N_side, V_com = 64, 3.**3.                  #Number of cells/side of sim volume & cell volume
    base_path = './NN_Training/Deltas/Delta_'                   #Base path to trained models 
    save_path = './Results/Box_' + str(N_side) + special + '/'  #And path to save results
    try:
        os.mkdir(save_path)     #Create results directory
    except OSError:
        print(OSError)

    z = np.linspace(15., 60., 901)              #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))          #Hubble time array in years
    N_EPOCHS_III, N_EPOCHS_II = 10000, 1000     #Number of epochs each population's NN model was trained over
    vbcs = 30.* np.linspace(0.0, 3.0, 301)      #Streaming velocities in km/s
    deltas = np.load('./Delta_Bins_64_400.npy') #Array of binned overdensity values
    J_0 = np.load('./J_LW_0/J_0/All_J_0.npy')   #M_crit(J_LW=0) values for all vbc bins
    MMHs = np.load('./z_arrays/All_MMHs.npy')               #2D array of all MMH(z) arrays 
    all_delta = np.load('./Overdensity_Field_64_192.npy')   #Load all z=15 overdensities 
    all_vbc = np.abs(np.load('./Vbc_Field_64_192.npy'))     #And vbc values (at z = 1060)
    all_delta_i = np.load('./delta_i.npy')                  #As well as the delta array indicies of each cell
    all_vbc_i = np.load('./vbc_i.npy')                                  #And streaming velocity indicies everywhere
    all_f_LW = np.load('./All_f_LW.npy', allow_pickle=True)             #Attenuation value arrays for each distance at each z (Get_J)
    dists_LW = np.load('./LW_Smoothing_Scales.npy', allow_pickle=True)  #Distances (cMpc) over which SFR cube is smoothed at each z step
    dists_z = np.load('./LW_Smoothing_zs.npy', allow_pickle=True)       #Redshifts at those constant distances

    SFR_III = np.zeros((N_side, N_side, N_side, len(z)))            #Initialize 4D arrays that are the size of our 3D cube with each element...
    SFR_II = np.zeros((N_side, N_side, N_side, len(z)))             #Being a z-length array of PopII/III SFRs to populate as we step through time
    Pop_III = np.zeros((N_side, N_side, N_side, len(z)))            #We will get these SFR values from the emulated PopIII stellar mass via...
    Pop_III_Smooth = np.zeros((N_side, N_side, N_side, len(z)))     #The smoothed M_star array for calculating SFR_III...
    Pop_II_Smooth = np.zeros((N_side, N_side, N_side, len(z)))      #And smoothed PopII array for SFR_II
    Pop_II_B = np.zeros((N_side, N_side, N_side, len(z)))           #And the emulated PopII stellar masses
    Pop_II_S = np.zeros((N_side, N_side, N_side, len(z)))           #For both steady & bursty PopII SF
    J_z = np.zeros((N_side, N_side, N_side, len(z)))                #Initialize a cube to record both the LW background...
    M_crit_z = np.zeros((N_side, N_side, N_side, len(z)))           #And critical masses in all cells over time
    t_III_all = np.full((N_side, N_side, N_side), fill_value=1e12)  #Also an array of PopIII turn on times which, after t_delay, allows PopII SF

    coeffs = np.load('./Schaerer_PopIII_Fit_Coeffs.npy', allow_pickle=True) #Fitting coefficients for the PopIII stellar lifetime...
    coeffs_life, coeffs_ion, coeffs_LW = coeffs[0], coeffs[1], coeffs[2]    #The ionizing/LW photon production rates [s^-1] 
    coeffs_eLW, coeffs_eta = coeffs[3], coeffs[4]                           #And # of ion/LW photons per baryon 
    fit_life, fit_ion, fit_LW, fit_eLW, fit_eta = 0., 0., 0., 0., 0.        #Initialize fit value for each property
    for i in range(0, len(coeffs_life)):
        fit_life += coeffs_life[i]*np.log10(star_m0)**float(len(coeffs_life)-i-1)   #Calculate fit values for each property
    for i in range(0, len(coeffs_ion)):
        fit_ion += coeffs_ion[i]*np.log10(star_m0)**float(len(coeffs_ion)-i-1)
    for i in range(0, len(coeffs_LW)):
        fit_LW += coeffs_LW[i]*np.log10(star_m0)**float(len(coeffs_LW)-i-1)
    for i in range(0, len(coeffs_eLW)):
        fit_eLW += coeffs_eLW[i]*np.log10(star_m0)**float(len(coeffs_eLW)-i-1)
    for i in range(0, len(coeffs_eta)):
        fit_eta += coeffs_eta[i]*np.log10(star_m0)**float(len(coeffs_eta)-i-1)
    print('(Log) Lifetime, Ionizing, LW = ', fit_life, fit_ion, fit_LW)
    print('eta_LW, eta_ion: ', fit_eLW, fit_eta)
    fit_life, fit_ion, fit_LW, fit_eLW, fit_eta = 10.**(fit_life), 10.**(fit_ion), 10.**(fit_LW), 10.**(fit_eLW), 10.**(fit_eta)    #Raise fit to 10^fit 

    H_0, Omega_m, c = 67., 0.32, 29979245800.                       #Hubble constant in km/s/Mpc, cosmological matter density parameter, speed of light (cm/s)
    eta_II, eta_III, H_0_s = 9690., fit_eLW, H_0/3.0857e19          #LW/ionizing photons per baryon by stellar population & Hubble constant in units of 1/s
    m_proton, h_planck, h = 1.673e-24, 6.626e-27, H_0/100.          #Mass of a proton in g & Planck's Constant in cgs & Reduced Hubble constant
    E_LW_l_eV, E_LW_u_eV, Omega_L = 11.2, 13.6, 1.-Omega_m          #LW energy limits in eV, DE cosmo density parameter
    E_LW_l, E_LW_u = E_LW_l_eV*1.6022e-12, E_LW_u_eV*1.6022e-12     #LW energy limits in erg
    E_LW = (E_LW_u+E_LW_l)/2.                                       #Average energy of LW in erg
    nu_l, nu_u = E_LW_l/h_planck, E_LW_u/h_planck                   #Frequencies of LW energy limits
    dnu_LW, T_vir_steady = nu_u - nu_l, 1.76e4                      #Difference of frequency limits & virial T threshold for steady PopII SF
    C_emis = (1./m_proton) * E_LW / dnu_LW                          #Emissivity coefficient in calculating epsilon_i
    LW_vals = np.array([H_0_s,Omega_m,eta_II,eta_III,c,C_emis])     #Values needed to run Get_J (avoid these calculations every z step)

    # Below is the main code for simulating PopII/III star formation in a large-scale 3D volume using neural networks ----------------
    for iz in range(899, -1, -1):                       #Step through time and begin simulating
        z_pres = round(z[iz], 2)                        #Current redshift step
        dt, t_delay_ago = t[iz]-t[iz+1], t[iz]-t_delay  #Years since last step and t_H a delay time ago
        delay_iz = np.argmin(np.abs(t-t_delay_ago))     #Corresponding iz index of delay time ago
        if np.sum(Pop_III) < 1e-15:
            J = np.full((N_side, N_side, N_side), fill_value=1e-6)          #Before any SF, skip smoothing/Get_J and set J = 1e-6 J_21
        else:
            iz_stop = np.argmin(np.abs(z-dists_z[iz][-1])) + 5              #Redshift index at which Smooth_SFRs should stop (plus some buffer for 3 Myr smoothing)
            J = Get_J(z_pres, z, t, all_f_LW[iz], N_side, LW_vals, np.copy(SFR_III), SFR_II, iz, save_path, dists_LW, np.array(dists_z[iz]), iz_stop)
        M_a = 5.4e7*(((1.+z_pres)/11.)**-1.5)                                           #Calculate atomic cooling threshold for M_crit calc
        M_crit = Get_M_Crit(J, all_vbc, z_pres, J_0[:,iz], all_vbc_i, M_a, M_crit_mod)  #Critical mass caluclation
        J_z[:,:,:,iz], M_crit_z[:,:,:,iz] = J, M_crit                                   #Record values of J_LW & M_Crit in grand arrays
        print('z, J_LW, M_crit: ', z_pres, np.mean(J), np.mean(M_crit))
        Omega_z_m = (Omega_m*((1.+z_pres)**3.)) / (Omega_m*((1.+z_pres)**3.) + Omega_L) #EQ 23 from Barkana & Loeb 2001
        del_c = 18.*(np.pi**2.) + 82.*(Omega_z_m-1.) - 39.*((Omega_z_m-1.)**2.)         #EQ 22 "" -- Both used in getting T_vir for halos
        for delta_i in range(0, len(deltas)):                       #Loop through each overdensity bin and emulate SF in all cells with that delta value
            emulating = base_path + str(delta_i) + '/'              #Path to this bin's NN emulation model
            model_III = SimpleNet3().to(device=device)              #Set up trained NN emulation for PopIII SF
            model_III.load_state_dict(torch.load(emulating + '/Emul_' + str(N_EPOCHS_III) + '_PopIII' + NN_special + '.pt'))
            model_III.eval()
            model_IIB = SimpleNet2b().to(device=device)             #And set up trained NN emulation for both bursty......
            model_IIB.load_state_dict(torch.load(emulating + '/Emul_' + str(N_EPOCHS_II) + '_PopII' + NN_special + '_Bursty.pt'))
            model_IIB.eval()
            try:
                model_IIS = SimpleNet2s().to(device=device)         #.....and steady PopII star formation
                model_IIS.load_state_dict(torch.load(emulating + '/Emul_' + str(N_EPOCHS_II) + '_PopII' + NN_special + '_Steady.pt'))
                model_IIS.eval()
            except:
                pass
            MMH = MMHs[delta_i]                                     #Relevant MMH array for this overdensity bin merger history -- Calculate current T_vir
            T_vir = 1.98e4 * (MMH[iz]/(1e8/h))**(2./3.) * ((Omega_m*del_c)/(18.*(np.pi**2.)*Omega_z_m))**(1./3.) * ((1.+z_pres)/10.)
            cells = np.array(np.where(all_delta_i == delta_i))      #Coordinates of cells within overall volume within this overdensity bin
            for i in range(0, len(cells[0])):                       #Loop through all of those cells in this bin
                ii, ij, ik = cells[0][i], cells[1][i], cells[2][i]  #Cooridantes for this cell
                # The following block of code is for calculating PopIII star formation --------------------------------------------------
                if MMH[iz] >= M_crit[ii,ij,ik] or t_III_all[ii,ij,ik] < 9e11:                       #If this cell's MMH > M_crit or it already has PopIII, emulate
                    M_crit_cell, J_LW_cell = M_crit_z[ii,ij,ik,iz+1:], J_z[ii,ij,ik,iz+1:]          #This cell's M_crit & J_LW histories up to previous step
                    int_M, int_J = np.trapz(M_crit_cell, z[iz+1:]), np.trapz(J_LW_cell, z[iz+1:])   #Integral of M_crit & J_LW histories up to previous step
                    log_M, log_J = np.log10(int_M), np.log10(int_J)                                 #Take their logs to limit dynamic range
                    vbc = all_vbc[ii,ij,ik] * ((1.+z_pres)/1060.)                                   #Get current vbc(z) value from overall array
                    log_J_LW, log_MMH = np.log10(J[ii,ij,ik]), np.log10(MMH[iz])                    #Get current J_LW, M_crit, MMH & take their logs too
                    inputs_III_0 = np.array((log_M, log_J, vbc, log_J_LW, log_MMH))                 #PopIII emulation input array
                    inputs_III = torch.tensor(inputs_III_0)                                                                             #Put array into PyTorch tensor
                    Pop_III[ii,ij,ik,iz] = max(10.**(model_III(inputs_III.float())), Pop_III[ii,ij,ik,iz+1])                            #Emulate & save M_star(z) (don't let it be < last z)
                    Pop_III_Smooth[ii,ij,ik,iz] = np.mean(Pop_III[ii,ij,ik,iz:iz+10])                                                   #Record smoothed M_star value by averaging over last dz=0.5
                    SFR_III[ii,ij,ik,iz] = 200.*(np.floor(Pop_III_Smooth[ii,ij,ik,iz])-np.floor(Pop_III_Smooth[ii,ij,ik,iz+1])) / dt    #Caluclate SFR (round PopIII down to account for M_star smoothing)
                    if Pop_III_Smooth[ii,ij,ik,iz] < 1.0 and Pop_III_Smooth[ii,ij,ik,iz] > 1e-10:                                       #Make sure early PopIII M_star < 1 isn't missed
                        SFR_III[ii,ij,ik,iz] = 200.*(Pop_III_Smooth[ii,ij,ik,iz]-Pop_III_Smooth[ii,ij,ik,iz+1]) / dt    #I.e. don't round down to preserve PopIII onset timing
                    if t_III_all[ii,ij,ik] == 1e12:                                                                     #If this is the first PopIII SF event...
                        t_III_all[ii,ij,ik] = t[iz]                                                                     #Record t_onset to later allow for PopII SF after t_delay
                # The following block of code is for calculating BURSTY PopII star formation, above is PopIII --------------------------------------------
                if t[iz] > (t_III_all[ii,ij,ik] + t_delay):                                     #After the delay time has passed, emulate PopII as well
                    M_crit_cell = M_crit_z[ii,ij,ik]                                            #Current cell's M_crit history
                    Mcrit_delay = M_crit_cell[delay_iz]                                         #And it's value one delay time ago
                    int_N_delay = np.abs(np.trapz(Pop_III[ii,ij,ik,delay_iz:], t[delay_iz:]))   #As well as the integral of PopIII up to t_delay ago
                    log_N_t = np.nan_to_num(np.log10(int_N_delay), neginf=-15)                  #Take logs to limit dynamic range 
                    log_M_t = np.nan_to_num(np.log10(Mcrit_delay), neginf=-15)
                    inputs_II_0 = np.array((log_N_t, log_M_t))                                  #Array of input parameters
                    inputs_II = torch.tensor(inputs_II_0)                                       #Put into PyTorch tensor form
                    Pop_II_B[ii,ij,ik,iz] = max(10.**(model_IIB(inputs_II.float())), Pop_II_B[ii,ij,ik,iz+1])   #Emulate & record star formation (don't let cumulative SF mass drop)
                    Pop_II_Smooth[ii,ij,ik,iz] += np.mean(Pop_II_B[ii,ij,ik,iz:iz+10])                          #Record smoothed value by averaging over last dz=0.5 (bursty)
                    # The following block of code is for calculating STEADY PopII star formation, above is BURSTY PopII --------------------------------------
#                    if t[iz] > (t_III_all[ii,ij,ik] + t_delay) and T_vir >= T_vir_steady:       #Also once the MMH has T_vir>T_vir_steady, emulate steady state PopII as well
                    if (t[iz] > (t_III_all[ii,ij,ik] + t_delay) and T_vir >= T_vir_steady) or (t[iz] > (t_III_all[ii,ij,ik] + t_delay) and Pop_II_S[ii,ij,ik,iz+1] > 1e-10):
                        inputs_II_0 = np.array((log_N_t, log_M_t, np.log10(MMH[iz])))           #Array of input parameters (already known)
                        inputs_II = torch.tensor(inputs_II_0)                                                     #Put into PyTorch tensor form
                        Pop_II_S[ii,ij,ik,iz] = max(10.**(model_IIS(inputs_II.float())), Pop_II_S[ii,ij,ik,iz+1]) #Emulate & record star formation (again, don't let it drop)
                        Pop_II_Smooth[ii,ij,ik,iz] += np.mean(Pop_II_S[ii,ij,ik,iz:iz+10])                                      #Add steady SF to smoothed SFR array
                    SFR_II[ii,ij,ik,iz] = (np.floor(Pop_II_Smooth[ii,ij,ik,iz])-np.floor(Pop_II_Smooth[ii,ij,ik,iz+1])) / dt    #Finally calculate PopII SFR
        # Print out results every time step, and overwrite previous result arrays every dz = 0.5 -----------------------------------------
        print('Mean M_stellar (PopIII/IIb/IIs): ', np.mean(Pop_III[:,:,:,iz]), np.mean(Pop_II_B[:,:,:,iz]), np.mean(Pop_II_S[:,:,:,iz]))
        print('Mean SFR III/II: ', np.mean(SFR_III[:,:,:,iz]), np.mean(SFR_II[:,:,:,iz]))
        if iz % 100 == 0:
            np.save(save_path + 'SFR_II_all.npy', SFR_II)
            np.save(save_path + 'SFR_III_all.npy', SFR_III)
            np.save(save_path + 'Mstar_II_B_all.npy', Pop_II_B)
            np.save(save_path + 'Mstar_II_S_all.npy', Pop_II_S)
            np.save(save_path + 'Mstar_III_all.npy', Pop_III)
            np.save(save_path + 'J_z_all.npy', J_z)
            np.save(save_path + 'M_crit_all.npy', M_crit_z)
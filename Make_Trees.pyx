import numpy as np
from scipy import special, interpolate
import scipy.integrate, os, time, shutil
cimport numpy as np
cimport cython
np.set_printoptions(suppress=True)
 
def get_sigma(double Mass, np.ndarray[np.float64_t,ndim=1] masses, np.ndarray[np.float64_t,ndim=1] sigmas):
    return np.interp(np.log(Mass), np.log(masses), sigmas)

def get_mass(double sigma, np.ndarray[np.float64_t,ndim=1] masses, np.ndarray[np.float64_t,ndim=1] sigmas):
    return np.exp(np.interp(sigma, np.flip(sigmas), np.flip(np.log(masses))))

def get_Trees(double M_start, np.ndarray[np.float64_t,ndim=1] z, np.ndarray[np.float64_t,ndim=1] sigmas, np.ndarray[np.float64_t,ndim=1] D, np.ndarray[np.float64_t,ndim=1] masses, np.ndarray[np.float64_t,ndim=1] params):
    cdef int iz, ih
    cdef double d_crit, Sres, M_crit_keep, z_pres, last_z, Sprev, x, Stemp, Mtemp
    cdef np.ndarray[np.float64_t,ndim=1] prev_halo_m, prev_halo_ID, new_row1, new_row2
    cdef np.ndarray[np.float64_t,ndim=2] all_halos

    d_crit, Sres, M_crit_keep = params[0], params[1], params[2]     #Necessary parameters
    all_halos = np.array([[0, z[0], M_start, -1]])                  #Initialize first row of tree array: ID, z, mass, & ID_desc
    for iz in range(1, len(z)):                                     #Now loop through z array building merger history
        z_pres, last_z = round(z[iz], 2), round(z[iz-1], 2)         #Current & previous redshift steps
        print(z_pres)
        all_halo_z = all_halos[:,1]                                       #Vector of all redshifts from overall array
        prev_halo_m = all_halos[np.abs(all_halo_z - last_z) < 1e-8, 2]    #Vector of previous halo masses
        prev_halo_ID = all_halos[np.abs(all_halo_z - last_z) < 1e-8, 0]   #Vector of previous halo IDs
        for ih in range(len(prev_halo_m)):                                #Loop through all present halos
            if prev_halo_m[ih] < M_crit_keep:                             #Make sure that it is large enough to have progenitors
                continue
            Sprev, x = get_sigma(prev_halo_m[ih], masses, sigmas), np.random.rand()                     #Get sigma from it's mass & draw a random number from 0-1
            Stemp = np.sqrt(((d_crit*(1./D[iz]-1./D[iz-1])/special.erfinv(x))**2.)/2. + Sprev**2.)      #EQ 5 from Visbal et al. 2014 solved for Sigma(M_1)
            if (Stemp > Sres):
                new_row1 = np.array([np.size(all_halos,0), z_pres, prev_halo_m[ih], prev_halo_ID[ih]])  #If progenitor sigma is too large, no progenitor halo
                all_halos = np.row_stack((all_halos, new_row1))                                         #Append array with same halo
            else:
                Mtemp = get_mass(Stemp, masses, sigmas)             #If resolution is sufficient, get the halo's mass & append array with each
                new_row1 = np.array([np.size(all_halos,0), z_pres, max(Mtemp,prev_halo_m[ih]-Mtemp), prev_halo_ID[ih]])
                new_row2 = np.array([np.size(all_halos,0)+1, z_pres, min(Mtemp,prev_halo_m[ih]-Mtemp), prev_halo_ID[ih]])
                all_halos = np.row_stack((all_halos, new_row1))     #Append with larger progenitor of the previous halo
                all_halos = np.row_stack((all_halos, new_row2))     #And append with smaller progenitor
    return all_halos

def Get_All(int im, int iN):            #Main code for generating trees, generates tree for mass bin im with ID number iN
    cdef double d_crit, M_res, M_z20, M_crit_keep, Sres, start, end
    cdef np.ndarray[np.float64_t,ndim=1] z, z_ff, sigmas, D, mass_bins, masses, params
    cdef np.ndarray[np.float64_t,ndim=2] all_halos

    start = time.time()                 #Record start time
    z = np.linspace(15., 60., 901)      #Redshift array
    z_ff = np.load('./z_ff_15.npy')     #FF-time-ago redshift array
    sigmas = np.load('./Sigmas.npy')    #Load sigmas values
    D = np.load('./Dz_15.npy')              #Growth factor at all z
    mass_bins = np.logspace(5.6, 9.5, 40)   #Array of halo mass bins

    masses = np.logspace(0., 16.1, 1000)                          #Array of sample masses well beyond the max/min bins used
    d_crit, M_res, M_z20, M_crit_keep = 1.686, 200., 2.05e5, 5e4  #Various parameters for tree code
    Sres = get_sigma(M_res, masses, sigmas)                       #Variance for minimum halo mass
    params = np.array([d_crit, Sres, M_crit_keep])                #Array of values to pass to tree code

    all_halos = get_Trees(mass_bins[im], z, sigmas, D, masses, params)      #Generate & save merger tree
    np.save('./All_Merger_Trees/m' + str(im) + '/m' + str(im) + '_' + str(iN) + '_Merger_Tree.npy', all_halos, allow_pickle=True)
    end = time.time() - start 
    print('Runtime (min): ', round(end/60.,2))

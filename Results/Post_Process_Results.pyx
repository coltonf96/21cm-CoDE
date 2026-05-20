import numpy as np
import os
cimport numpy as np
cimport cython

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

def Smooth(np.ndarray[np.float64_t,ndim=2] SF, np.ndarray[np.float64_t,ndim=1] z):
    cdef int iz, ii, iz_smooth
    cdef double M_pres, dM, dM_dz
    cdef np.ndarray[np.float64_t,ndim=1] SF_II, SF_dots, M_star, M_dots, M_smooth, SF_smooth 

    SF_II, SF_dots = SF[0,:], np.zeros(len(z))  #Isolate PopII SF data & initialize array to record SF(z) values > SF(z+dz)
    M_star, M_dots = SF[1,:], np.zeros(len(z))  #Repeat for PopIII as in original simulation
    for iz in range(0, len(z)-1):                   #Step through time
        if M_star[iz] > M_star[iz+1] or iz == 0:    #If this M_star(z) is > than the previous M_star(z+dz)...
            M_dots[iz] = M_star[iz]                 #Record it in M_dots, otherwise it remains 0
        if SF_II[iz] > SF_II[iz+1] or iz == 0:      #Similarly, for PopII, if it's higher than z+dz, record
            SF_dots[iz] = SF_II[iz]
    M_smooth, SF_smooth = np.zeros(len(z)), np.zeros(len(z))  #Now initialize smoothed stellar mass array for both
    for iz in range(0, len(z)):                 #Step through time to populate M_smooth
        if M_dots[iz] > 1e-15:                  #Only when M_dots has a nonzero value...
            M_pres = M_dots[iz]                 #Record it and begin the smoothing loop
            for ii in range(1, len(z)-iz-1):    #Step through, at most, the rest of the z array 
                if M_dots[iz+ii] > 1e-15:       #Find the next z where M_dot(z) is nonzero...
                    iz_smooth = ii              #Record it & break out of the for loop
                    break
                if sum(M_dots[iz+ii:]) < 1e-15: #Whenever we reach the last dot...
                    iz_smooth = 1               #Use dz = 0.05 as the last iz_smooth value & break out of loop
                    break
            dM = M_pres - M_dots[iz+iz_smooth]  #Now determine how much M_star changes from M_pres to the next nonzero value
            dM_dz = dM/iz_smooth                #Divide by # of smoothing steps to get smoothing value for each z step
            for ii in range(0, iz_smooth):              #Now step back through z array to that next nonzero value
                    M_smooth[iz+ii] = M_pres-ii*dM_dz   #And populate M_smooth with step_size*N_step
        if SF_dots[iz] > 1e-15:                 #Repeat above 'if' statement for PopII
            M_pres = SF_dots[iz]
            for ii in range(1, len(z)-iz-1):    #Step through, at most, the rest of the z array 
                if SF_dots[iz+ii] > 1e-15:      #Find the next z where M_dot(z) is nonzero...
                    iz_smooth = ii              #Record it & break out of the for loop
                    break
                if sum(SF_dots[iz+ii:]) < 1e-15:    #Whenever we reach the last dot...
                    iz_smooth = 1                   #Use dz = 0.05 as the last iz_smooth value & break out of loop
                    break
            dM = M_pres - SF_dots[iz+iz_smooth]     #Now determine how much M_star changes from M_pres to the next nonzero value
            dM_dz = dM/iz_smooth                    #Divide by # of smoothing steps to get smoothing value for each z step
            for ii in range(0, iz_smooth):          #Now step back through z array to that next nonzero value
                SF_smooth[iz+ii] = M_pres-ii*dM_dz  #And populate SF_smooth with step_size*N_step
    M_smooth = np.nan_to_num(M_smooth)              #Remove NaNs from smoothed SF array 
    SF_smooth = np.nan_to_num(SF_smooth)            #For both stellar populations -- Overwrite SF file with smoothed values
    return(np.row_stack((SF_II, SF_smooth, M_star, M_smooth)))

def Smooth2(np.ndarray[np.float64_t,ndim=2] SF, np.ndarray[np.float64_t,ndim=1] z):
    cdef int iz, ii, iz_smooth
    cdef double M_pres, dM, dM_dz
    cdef np.ndarray[np.float64_t,ndim=1] M_burst, M_steady, M_star, M_b_dots, M_s_dots, M_dots, M_b_smooth, M_s_smooth, M_smooth

    M_burst, M_steady, M_star = SF[0,:], SF[1,:], SF[2,:]       #Isolate each row of data; bursty PopII, steady PopII, PopIII
    M_burst, M_steady = np.flipud(np.cumsum(np.flipud(M_burst))), np.flipud(np.cumsum(np.flipud(M_steady)))   #Get cumulative values for PopII
    M_b_dots, M_s_dots, M_dots = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z)) #Initialize array to record where SF(z) > SF(z+dz)
    for iz in range(0, len(z)-1):                       #Now step through time to populate dots arrays
        if M_burst[iz] > M_burst[iz+1] or iz == 0:      #If this bursty M_star(z) is > than the previous M_star(z+dz)...
            M_b_dots[iz] = M_burst[iz]                  #Record it in M_dots, otherwise it remains 0
        if M_steady[iz] > M_steady[iz+1] or iz == 0:    #Repeat for other rows (steady SF)
            M_s_dots[iz] = M_steady[iz]
        if M_star[iz] > M_star[iz+1] or iz == 0:        #And for PopIII SF
            M_dots[iz] = M_star[iz] 
    M_b_smooth, M_s_smooth, M_smooth = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z))   #Initialize smoothed SF arrays
    for iz in range(0, len(z)):                 #Step through time to populate M_smooth
        if M_dots[iz] > 1e-15:                  #Only when M_dots has a nonzero value...
            M_pres = M_dots[iz]                 #Record it and begin the smoothing loop
            for ii in range(1, len(z)-iz-1):    #Step through, at most, the rest of the z array 
                if M_dots[iz+ii] > 1e-15:       #Find the next z where M_dot(z) is nonzero...
                    iz_smooth = ii              #Record it & break out of the for loop
                    break
                if sum(M_dots[iz+ii:]) < 1e-15: #Whenever we reach the last dot...
                    iz_smooth = 1               #Use dz = 0.05 as the last iz_smooth value & break out of loop
                    break
            dM = M_pres - M_dots[iz+iz_smooth]      #Now determine how much M_star changes from M_pres to the next nonzero value
            dM_dz = dM/iz_smooth                    #Divide by # of smoothing steps to get smoothing value for each z step
            for ii in range(0, iz_smooth):          #Now step back through z array to that next nonzero value
                M_smooth[iz+ii] = M_pres - ii*dM_dz #And populate M_smooth with step_size*N_step
        if M_b_dots[iz] > 1e-15:                    #Only when M_dots has a nonzero value...
            M_pres = M_b_dots[iz]                   #Record it and begin the smoothing loop
            for ii in range(1, len(z)-iz-1):        #Step through, at most, the rest of the z array 
                if M_b_dots[iz+ii] > 1e-15:         #Find the next z where M_dot(z) is nonzero...
                    iz_smooth = ii                  #Record it & break out of the for loop
                    break
                if sum(M_b_dots[iz+ii:]) < 1e-15:   #Whenever we reach the last dot...
                    iz_smooth = 1                   #Use dz = 0.05 as the last iz_smooth value & break out of loop
                    break
            dM = M_pres - M_b_dots[iz+iz_smooth]    #Now determine how much M_star changes from M_pres to the next nonzero value
            dM_dz = dM/iz_smooth                    #Divide by # of smoothing steps to get smoothing value for each z step
            for ii in range(0, iz_smooth):              #Now step back through z array to that next nonzero value
                M_b_smooth[iz+ii] = M_pres - ii*dM_dz   #And populate M_smooth with step_size*N_step
        if M_s_dots[iz] > 1e-15:                        #Only when M_dots has a nonzero value...
            M_pres = M_s_dots[iz]                   #Record it and begin the smoothing loop
            for ii in range(1, len(z)-iz-1):        #Step through, at most, the rest of the z array 
                if M_s_dots[iz+ii] > 1e-15:         #Find the next z where M_dot(z) is nonzero...
                    iz_smooth = ii                  #Record it & break out of the for loop
                    break
                if sum(M_s_dots[iz+ii:]) < 1e-15:   #Whenever we reach the last dot...
                    iz_smooth = 1                   #Use dz = 0.05 as the last iz_smooth value & break out of loop
                    break
            dM = M_pres - M_s_dots[iz+iz_smooth]    #Now determine how much M_star changes from M_pres to the next nonzero value
            dM_dz = dM/iz_smooth                    #Divide by # of smoothing steps to get smoothing value for each z step
            for ii in range(0, iz_smooth):              #Now step back through z array to that next nonzero value
                M_s_smooth[iz+ii] = M_pres - ii*dM_dz   #And populate M_smooth with step_size*N_step
    M_smooth = np.nan_to_num(M_smooth)                    #Remove NaNs from smoothed SF array 
    M_b_smooth = np.nan_to_num(M_b_smooth)            #For both stellar populations & both PopII
    M_s_smooth = np.nan_to_num(M_s_smooth)            #Overwrite SF file with smoothed values
    return(np.row_stack((M_burst, M_b_smooth, M_steady, M_s_smooth, M_star, M_smooth)))

def Process(int sim, str special, np.ndarray[np.int_t,ndim=1] processes, int overwrite=0):
    cdef int N_side, slice_n, i, x_i, y_i, z_i, iz_II, iz_III, iz_II_S, iz_II_B, iz, L_side
    cdef str path, save_path, SAM_path, suffix
    cdef double V_com, dt, c, Ly_a
    cdef np.ndarray[np.int_t,ndim=1] sample_delta_i, sample_vbc_i, coords
    cdef np.ndarray[np.int_t,ndim=2] samples, all_coords
    cdef np.ndarray[np.int_t,ndim=3] all_delta_i, all_vbc_i
    cdef np.ndarray[np.float64_t,ndim=1] z, t, sample_vbc, M_II, M_III, M_II_e, M_III_e, avg_SFRD_II, SD_SFRD_II, avg_SFRD_III, SD_SFRD_III, avg_J, SD_J, avg_crit, SD_crit, avg_II, SD_II, avg_II_B, SD_II_B, avg_II_S, SD_II_S, avg_III, SD_III, Pk_II, k_II, nk_II, Pk_III, k_III, nk_III
    cdef np.ndarray[np.float64_t,ndim=2] sample_J, sample_M, SF, smoothed, all_pd_II, all_pd_III, all_pd_II_B, all_pd_II_S
    cdef np.ndarray[np.float64_t,ndim=3] all_delta, all_vbc, SFRD_II, SFRD_III, J, M_crit
    cdef np.ndarray[np.float64_t,ndim=4] J_z_all, M_crit_z, M_II_all, M_III_all, M_II_B_all, M_II_S_all, SFR_II_all, SFR_III_all

    z = np.linspace(15., 60., 901)      #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))  #Hubble time array
    N_side, V_com = 64, 27.             #Cells/side of sim volume & cell volume [cMpc^3]
    slice_n = int(N_side/2)             #Middle slice of cube
    if sim == 0 or sim == 2:
        path = './Box_' + str(N_side) + special + '/'       #Path where all box results are saved for fid/bursty PopII models
    else:
        path = './Integral_' + str(N_side) + special + '/'  #Or path to all HMF integral method results
    save_path = path + '/Testing/'                          #Path to save samplings of this run
    SAM_path = save_path + 'SAM_Results/'                   #Path to random sample SAM data
    if processes[0] == 1 or processes[4] == 1:                  #For getting random samples & power spectra...
        all_delta = np.load('../Overdensity_Field_64_192.npy')  #Load in overdensity field
        all_vbc = np.abs(np.load('../Vbc_Field_64_192.npy'))    #And vbc values

    if processes[0] == 1:           #Randomly sample 100 cells from completed run
        try:
            os.mkdir(save_path)     #Make sure we have a place to save random sample data
        except OSError:
            print(OSError)
        all_delta_i = np.load('../delta_i.npy').astype(int)     #Load in overdensity bin indicies
        all_vbc_i = np.load('../vbc_i.npy').astype(int)         #And the same for vbc
        J_z_all = np.load(path + 'J_z_all.npy')             #Load in LW background from full simulation for this cell
        M_crit_z = np.load(path + 'M_crit_all.npy')         #And critical mass
        samples, sample_vbc = np.zeros((100, 3),dtype=int), np.zeros(100)                   #Initialize sample coords list & all properties of them
        sample_delta_i, sample_vbc_i = np.zeros(100).astype(int), np.zeros(100).astype(int) #Including vbc/delta, their IDs, J_LW & M_crits
        sample_J, sample_M = np.zeros((100, 901)), np.zeros((100, 901))
        for i in range(0, 100):                                                                     #Then loop through, sampling cell coordinates & their properties
            x_i, y_i, z_i = np.random.randint(0,N_side), np.random.randint(0,N_side), np.random.randint(0,N_side)   #Random x/y/z coords of cell
            samples[i], sample_vbc[i] = [x_i, y_i, z_i], all_vbc[x_i,y_i,z_i]                       #Update sample coords & vbcs
            sample_delta_i[i], sample_vbc_i[i] = all_delta_i[x_i,y_i,z_i], all_vbc_i[x_i,y_i,z_i]   #Record overdensity/vbc bins
            sample_J[i], sample_M[i] = J_z_all[x_i,y_i,z_i], M_crit_z[x_i,y_i,z_i]                  #And record J_LW(z)/M_crit(z)
        np.save(save_path + 'Random_Samples.npy', samples)
        np.save(save_path + 'RS_delta_i.npy', sample_delta_i)   #Save all sample data
        np.save(save_path + 'RS_vbc_i.npy', sample_vbc_i)
        np.save(save_path + 'RS_vbc.npy', sample_vbc)
        np.save(save_path + 'RS_J_z.npy', sample_J)
        np.save(save_path + 'RS_Mcrit_z.npy', sample_M)

# ------ AFTER RUNNING THE ABOVE BLOCK OF CODE WHICH RANDOMLY SAMPLES THE CELLS, RUN EACH THROUGH FULL SAM TO GET EXPECTED STAR FORMATION HISTORY BEFORE CONTINUING TO REST OF CODE ------

    if processes[1] == 1:
          print('Smoothing M_star from Random Samples')     #PopIII M_star smoothing of random samples
          all_coords = np.load(save_path + 'Random_Samples.npy').astype(int)    #Load in randomly sampled cells
          for i in range(0, len(all_coords)):               #Loop through all sample cells   
            coords = all_coords[i]                          #Current cell coordinates
            suffix = str(coords[0]) + '_' + str(coords[1]) + '_' + str(coords[2]) + '.npy'  #Use coords to generate unique file name suffix
            SF = np.load(SAM_path + 'SF_' + suffix)         #Load in SF results of this sample cell
            if (sim == 0 and len(SF) > 2.5) or (sim == 2 and len(SF) > 3.5):
                print('Already smoothed SF_', suffix, i)    #Make sure I haven't smoothed them already
                continue
            if sim == 0:
                smoothed = Smooth(SF, z)                    #For fiducial model, smooth both PopII/III M_star 
            elif sim == 2:
                smoothed = Smooth2(SF, z)                   #For bursty PopII model, smooth both PopII M_star arrays & PopIII M_star array
            np.save(SAM_path + 'SF_' + suffix, smoothed)    #And save

    if processes[2] == 1:
        print('Calculating sample % diffs')     #Get all percent differences from the sample cell runs
        try:
            os.mkdir(save_path + 'Figures/')    #Make path to save results for this run 
        except OSError:
            print(OSError)
        all_coords = np.load(save_path + 'Random_Samples.npy').astype(int)  #Load sampled coords
        if sim == 0:
            M_II_all = np.load(path + 'Mstar_II_all.npy')     #Fiducial emulated PopII cM_star histories
            M_III_all = np.load(path + 'Mstar_III_all.npy')   #And PopIII cM_star
            all_pd_II, all_pd_III = np.zeros((100,len(z))), np.zeros((100,len(z)))      #Initialize percent difference arrays
            for i in range(0, len(all_coords)):                                         #Loop through all of the runs  
                coords = all_coords[i]                                                  #This cell's cube coordinates
                suffix = str(coords[0]) + '_' + str(coords[1]) + '_' + str(coords[2])   #Suffix for SF arryas to be loaded in
                SF = np.load(SAM_path + '/SF_' + suffix + '.npy')                       #It's SF history from full SAM
                M_II, M_III = SF[1,:], SF[3,:]                                          #PopII & PopIII cM_star (smoothed)
                M_II_e = M_II_all[coords[0],coords[1],coords[2]]                        #PopII SF history from emulation cube
                M_III_e = M_III_all[coords[0],coords[1],coords[2]]                      #And emulated PopIII SF history
                iz_II, iz_III = np.nonzero(M_II)[0][-1], np.nonzero(M_III)[0][-1]       #SMD & M_star starting redshifts
                all_pd_II[i,:iz_II], all_pd_III[i,:iz_III] = np.abs((M_II[:iz_II]-M_II_e[:iz_II]))/M_II[:iz_II], np.abs((M_III[:iz_III]-M_III_e[:iz_III]))/M_III[:iz_III]
            np.save(save_path + '/RS_Perc_Diffs_II.npy', all_pd_II)
            np.save(save_path + '/RS_Perc_Diffs_III.npy', all_pd_III)   #Save all of them
        elif sim == 2:
            M_II_B_all = np.load(path + 'Mstar_II_B_all.npy')   #Bursty emulated PopII cM_star histories
            M_II_S_all = np.load(path + 'Mstar_II_S_all.npy')   #Steady ""
            M_III_all = np.load(path + 'Mstar_III_all.npy')     #And emulated PopIII cM_star -- Initialize percent difference arrays for each 
            all_pd_II_B, all_pd_II_S, all_pd_III = np.zeros((100,len(z))), np.zeros((100,len(z))), np.zeros((100,len(z)))
            for i in range(0, len(all_coords)):                                         #Loop through all of the runs  
                coords = all_coords[i]                                                  #This cell's cube coordinates
                suffix = str(coords[0]) + '_' + str(coords[1]) + '_' + str(coords[2])   #Suffix for SF arryas to be loaded in
                SF = np.load(SAM_path + '/SF_' + suffix + '.npy')                       #It's SF history from full SAM
                M_II_B, M_II_S, M_III = SF[1,:], SF[3,:], SF[5,:]                       #PopII & PopIII cM_star (smoothed)
                M_II_B_e = M_II_B_all[coords[0],coords[1],coords[2]]                    #Bursty PopII SF history from emulation
                M_II_S_e = M_II_S_all[coords[0],coords[1],coords[2]]                    #And steady PopII SF history                
                M_III_e = M_III_all[coords[0],coords[1],coords[2]]                      #And emulated PopIII SF history
                iz_II_B, iz_III = np.nonzero(M_II_B)[0][-1], np.nonzero(M_III)[0][-1]   #Onset z for PopIII & bursty PopII
                all_pd_II_B[i,:iz_II_B] = np.abs((M_II_B[:iz_II_B]-M_II_B_e[:iz_II_B]))/M_II_B[:iz_II_B]
                all_pd_III[i,:iz_III] = np.abs((M_III[:iz_III]-M_III_e[:iz_III]))/M_III[:iz_III]    #Calculate & record percent differneces as f(z)
                if np.sum(M_II_S) > 1e-15:                      #It's possible that cells have no steady PopII
                    iz_II_S = np.nonzero(M_II_S)[0][-1]         #If it does, continue calculation for steady PopII as well
                    all_pd_II_S[i,:iz_II_S] = np.abs((M_II_S[:iz_II_S]-M_II_S_e[:iz_II_S]))/M_II_S[:iz_II_S]
                else:                                           #If there's no steady PopII SF...
                    all_pd_II_S[i,:] = np.nan                   #Make this entry NaNs (will take nanmean/std in figure)
            np.save(save_path + '/RS_Perc_Diffs_II_B.npy', all_pd_II_B)
            np.save(save_path + '/RS_Perc_Diffs_II_S.npy', all_pd_II_S)
            np.save(save_path + '/RS_Perc_Diffs_III.npy', all_pd_III)   #Save all of them

    if processes[3] == 1:
        print('Averaging overall results')  #Average all results from full sim
        try:
            os.mkdir(path + 'Figures/') #Make directory in which we'll save each
        except OSError:
            print(OSError)
        #Starting with the PopII/III SFRs -------------------------------------------
        SFR_II_all = np.load(path + 'SFR_II_all.npy')/V_com
        SFR_III_all = np.load(path + 'SFR_III_all.npy')/V_com           #Load in SFRDs
        avg_SFRD_II, SD_SFRD_II = np.zeros(len(z)), np.zeros(len(z))    #And initialize both average & SD arrays
        avg_SFRD_III, SD_SFRD_III = np.zeros(len(z)), np.zeros(len(z))
        for iz in range(850, -1, -1):
            SFRD_II, SFRD_III = SFR_II_all[:,:,:,iz], SFR_III_all[:,:,:,iz]             #Loop through time -- SFRDs of all cells
            avg_SFRD_II[iz], SD_SFRD_II[iz] = np.mean(SFRD_II), np.std(SFRD_II)
            avg_SFRD_III[iz], SD_SFRD_III[iz] = np.mean(SFRD_III), np.std(SFRD_III)     #Then calculate their avgs/SDs
        np.save(path + 'Figures/SFRD_II_Avg_SD.npy', np.column_stack((avg_SFRD_II, SD_SFRD_II)))    #Save averages/SDs
        np.save(path + 'Figures/SFRD_III_Avg_SD.npy', np.column_stack((avg_SFRD_III, SD_SFRD_III)))
        del SFR_II_all, SFR_III_all     #Delete large memory arrays
        #Then the J_LW(z) and M_crit(z) ----------------------------------------------
        J_z_all = np.load(path + 'J_z_all.npy')
        M_crit_z = np.load(path + 'M_crit_all.npy')             #Now load in J_LW & M_Crit
        avg_J, SD_J = np.zeros(len(z)), np.zeros(len(z))        #Initialize their avgs/SDs
        avg_crit, SD_crit = np.zeros(len(z)), np.zeros(len(z))
        for iz in range(850, -1, -1):
            J, M_crit = J_z_all[:,:,:,iz], M_crit_z[:,:,:,iz]   #Isolate current z values for J_LW/M_crit
            avg_J[iz], SD_J[iz] = np.mean(J), np.std(J)         #Then calculate & record avgs/SDs
            avg_crit[iz], SD_crit[iz] = np.mean(M_crit), np.std(M_crit)
        np.save(path + 'Figures/J_LW_Avg_SD.npy', np.column_stack((avg_J, SD_J)))
        np.save(path + 'Figures/M_crit_Avg_SD.npy', np.column_stack((avg_crit, SD_crit)))
        del J_z_all, M_crit_z    #Delete large memory arrays
        #Finally, the PopII/III stellar masses ----------------------------------------
        if sim == 0 or sim == 1:
            if sim == 0:                                        #Now to average PopII stellar mass over time
                M_II_all = np.load(path + 'Mstar_II_all.npy')   #Fiducial & bursty PopII models save PopII stellar mass
            else:
                SFR_II_all = np.load(path + 'SFR_II_all.npy')   #Integral method only saves SFR
                M_II_all = np.zeros(np.shape(SFR_II_all))       #Initialize PopII M_star array for integral method
                for iz in range(0, len(z)-2):                   #Then loop through time 
                    dt = t[iz] - t[iz+1]                        #Change in Hubble time since last step
                    M_II_all[:,:,:,iz] = SFR_II_all[:,:,:,iz]*dt + M_II_all[:,:,:,iz+1]    #Calculate/record cumulative PopII M_star
            avg_II, SD_II = np.zeros(len(z)), np.zeros(len(z))  #Now initialize avg/SD arrays for PopII
            for iz in range(850, -1, -1):                       #Now step forward through time 
                avg_II[iz] = np.mean(M_II_all[:,:,:,iz])        #And get avgs/SDs of M_star,II(z)
                SD_II[iz] = np.std(M_II_all[:,:,:,iz])
            np.save(path + 'Figures/Mstar_II_Avg_SD.npy', np.column_stack((avg_II, SD_II))) #And save the data
            del M_II_all
        else:                                                       #And Bursty PopII method has two M_star arrays
            M_II_B_all, M_II_S_all = np.load(path + 'Mstar_II_B_all.npy'), np.load(path + 'Mstar_II_S_all.npy')
            avg_II, SD_II = np.zeros(len(z)), np.zeros(len(z))      #Initialize avg/SD arrays for the total PopII...
            avg_II_B, SD_II_B = np.zeros(len(z)), np.zeros(len(z))  #...for bursty PopII SF...
            avg_II_S, SD_II_S = np.zeros(len(z)), np.zeros(len(z))  #...and for steady PopII SF
            for iz in range(850, -1, -1):                           #Now step forward through time 
                avg_II[iz] = np.mean(M_II_B_all[:,:,:,iz] + M_II_S_all[:,:,:,iz])
                SD_II[iz] = np.std(M_II_B_all[:,:,:,iz] + M_II_S_all[:,:,:,iz])
                avg_II_B[iz] = np.mean(M_II_B_all[:,:,:,iz])        #And get avgs/SDs of M_star,II(z)
                SD_II_B[iz] = np.std(M_II_B_all[:,:,:,iz])
                avg_II_S[iz] = np.mean(M_II_S_all[:,:,:,iz])        #For total, bursty, & steady PopII SF
                SD_II_S[iz] = np.std(M_II_S_all[:,:,:,iz])
            np.save(path + 'Figures/Mstar_II_Avg_SD.npy', np.column_stack((avg_II, SD_II)))
            np.save(path + 'Figures/Mstar_II_B_Avg_SD.npy', np.column_stack((avg_II_B, SD_II_B))) #And save the data
            np.save(path + 'Figures/Mstar_II_S_Avg_SD.npy', np.column_stack((avg_II_S, SD_II_S)))
            del M_II_B_all, M_II_S_all
        #Above is PopII M_star, below is PopIII ---------------------------------------
        if sim == 0 or sim == 2:                                #And now to average PopIII stellar mass
            M_III_all = np.load(path + 'Mstar_III_all.npy')     #Fiducial & bursty PopII models save PopIII stellar mass
        elif sim == 1:
            SFR_III_all = np.load(path + 'SFR_III_all.npy')     #Integral method only saves SFR
            M_III_all = np.zeros(np.shape(SFR_III_all))         #Initialize PopII M_star array for integral method
            for iz in range(0, len(z)-2):                       #Then loop through time 
                dt = t[iz] - t[iz+1]                            #Change in Hubble time since last step
                M_III_all[:,:,:,iz] = SFR_III_all[:,:,:,iz]*dt + M_III_all[:,:,:,iz+1]    #Calculate/record cumulative PopIII M_star
        avg_III, SD_III = np.zeros(len(z)), np.zeros(len(z))    #Initialize avg/SD arrays for PopIII
        for iz in range(850, -1, -1):                           #Now step forward through time 
            avg_III[iz] = np.mean(M_III_all[:,:,:,iz])          #And get avgs/SDs of M_star,III(z)
            SD_III[iz] = np.std(M_III_all[:,:,:,iz])
        np.save(path + 'Figures/Mstar_III_Avg_SD.npy', np.column_stack((avg_III, SD_III))) #And save the data

    if processes[4] == 1:                   #Now to get the power spectra
        print('Getting Power Spectra')
        try:
            os.mkdir(path + 'Power_Spectra/')   #Make sure there is a directory to save PS to
        except OSError:
            print(OSError)
        SFR_II_all, SFR_III_all = np.load(path + 'SFR_II_all.npy'), np.load(path + 'SFR_III_all.npy')   #Load in SFRs 
        L_side = round(N_side*3.,1)         #Length of side of simulation volume (3 Mpc cells)
        c, Ly_a = 3e8, 1.216e-7             #Speed of light (m/s) & Lyman-alpha wavelength (m)
        for iz in range(0, len(z)):         #Loop through redshifts
            z_pres = round(z[iz],2)         #Current z value
            SFR_II, SFR_III = SFR_II_all[:,:,:,iz], SFR_III_all[:,:,:,iz]   #Current SFR(z) of all cells
            [Pk_II, k_II, nk_II] = Power(L_side, SFR_II, SFR_II, 50)        #Calculate the power spectra of each
            [Pk_III, k_III, nk_III] = Power(L_side, SFR_III, SFR_III, 50) 
            np.save(path + 'Power_Spectra/PS_SFR_II_' + str(round(z_pres,2)) + '.npy', np.row_stack((Pk_II, k_II, nk_II))) 
            np.save(path + 'Power_Spectra/PS_SFR_III_' + str(round(z_pres,2)) + '.npy', np.row_stack((Pk_III, k_III, nk_III)))

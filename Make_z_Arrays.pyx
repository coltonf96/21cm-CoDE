import numpy as np
import os, random
cimport numpy as np
cimport cython
 
def Get_Trees(int delta_i, str special=''):     # MAKE SURE THAT YOU RUN 'POISSON_SAMPLE.PYX' FIRST
    cdef int im, poisson_im, iN, max_z_i, iz, 
    cdef str array_path
    cdef list all_lengths, Grand_Trees, bin_trees, bin_lengths, all_trees
    cdef double max_z, z_pres
    cdef np.ndarray[np.int_t,ndim=1] tree_IDs, poisson
    cdef np.ndarray[np.float64_t,ndim=1] z, t, z_ff, mass_bins, deltas, Num_Halos, MMH, all_z, all_m
    cdef np.ndarray[np.float64_t,ndim=2] all_halos, pres_array

    z = np.linspace(15., 60., 901)          #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))      #Hubble time array
    z_ff = np.load('./z_ff_15.npy')         #Freefall time ago Redshift array
    mass_bins = np.logspace(5.6, 9.5, 40)               #Array of masses for each bin
    deltas = np.load('./21cm/Remade_Bins_64_400.npy')   #Array of overdensity bins

    Num_Halos = np.load('./HMFs/Num_Halos_' + str(delta_i) + '.npy')    #The HMF for this overdensity to be Poisson sampled
    array_path = './z_arrays/Trees_' + str(delta_i) + special           #Path to save all z_arrays
    poisson = np.load(array_path + '/Num_Trees.npy').astype(int)            #Array of Poisson-sampled N_trees(M)
    all_tree_IDs = np.load(array_path + '/Tree_IDs.npy', allow_pickle=True) #And the IDs of the merger trees used
    print(poisson)

    all_lengths, MMH = [], np.zeros(len(z)) #Initialize list of tree lengths & MMH(z) array
    Grand_Trees, max_z = [], 0.             #Initialize list of all merger trees & max z of tree halos
    for im in range(0, len(mass_bins)):     #Now loop through and get all of the trees needed to build the merger history
        tree_IDs = all_tree_IDs[im]         #This mass bins merger tree IDs
        bin_trees, bin_lengths = [], []     #Initialize lists of trees & lengths for this halo mass bin
        poisson_im = min(100, poisson[im])  #The number of Poisson-sampled merger trees for this bin
        for iN in range(0, poisson_im):     #Now loop through the relevant merger trees, load them in...
            all_halos = np.load('./All_Merger_Trees/m' + str(im) + '/m' + str(im) + '_' + str(tree_IDs[iN]) + '_Merger_Tree.npy', allow_pickle=True)
            bin_trees.append(all_halos)     #And append them onto grand list for z_array preparation
            all_z = all_halos[:,1]          #Isolate redshifts of all halos in tree
            bin_lengths.append(len(all_z))  #Then save the length of the merger tree to overall array
            max_z = max(max_z, all_z[-1])   #And find the max represented z for this tree, determine if it's larger than running max
        Grand_Trees.append(bin_trees)       #Append overall lists with mass bin lists
        all_lengths.append(bin_lengths)
    max_z_i = int(np.argmin(np.abs(z-max_z)))       #Save z_max index
    np.save(array_path + '/Tree_Lengths.npy', all_lengths, allow_pickle=True)   #Save the tree lengths
    np.save(array_path + '/z_start.npy', max_z_i)   #And the max z 
    print(all_lengths, max_z_i)

    for iz in range(0, max_z_i):                    #Now step through z & generate z-arrays to be used in simulation
        z_pres, all_trees = round(z[iz], 2), []     #Present z, initialize z_array for this redshift
        print(z_pres)
        for im in range(0, len(mass_bins)):             #Loop through trees, getting present arrays
            tree_IDs, bin_trees = all_tree_IDs[im], []  #This mass bins merger tree IDs & initialize mass bin z_array
            poisson_im = min(100, poisson[im])          #Number of Poisson-sampled trees for this mass bin
            for iN in range(0, poisson_im):             #Loop through the trees in this mass bin
                all_halos = Grand_Trees[im][iN]                 #Isolate relevant merger tree
                all_z, all_m = all_halos[:,1], all_halos[:,2]   #All halo redshifts & masses of the tree
                all_m = all_m[np.abs(all_z - z_pres) < 1e-8]    #Present halo masses
                if len(all_m) > 0.5:                            #Make sure there are halos at all
                    MMH[iz] = max(max(all_m), MMH[iz])              #Then assign most massive halo
                pres_array = all_halos[np.abs(all_z-z_pres) < 1e-8] #Relevant rows of merger tree array for present...
                bin_trees.append(pres_array)                        #Append the bin z_array
            all_trees.append(bin_trees)                         #Then after all bins, append overall array
        print(np.shape(all_trees))
        np.save(array_path + '/' + str(z_pres) + '_Arrays.npy', all_trees, allow_pickle=True) #Save this z_array
    np.save(array_path + '/MMH.npy', MMH)       #Save MMH(z) once finished
    Grand_Trees = []                            #Remove huge array of trees
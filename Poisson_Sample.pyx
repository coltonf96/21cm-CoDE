import numpy as np
import os, random
cimport numpy as np
cimport cython

def Poisson(str special=''):
    cdef int im, i, poisson_im
    cdef str array_path
    cdef list all_tree_IDs
    cdef np.ndarray[np.int_t,ndim=1] poisson, arange, N_Trees_M, tree_IDs, available
    cdef np.ndarray[np.float64_t,ndim=1] z, z_ff, t, mass_bins, deltas, Num_Halos, library
    cdef np.ndarray[np.float64_t,ndim=2] drawable
 
    z = np.linspace(15., 60., 901)          #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))      #Hubble time array
    z_ff = np.load('./z_ff_15.npy')         #Freefall time ago Redshift array
    mass_bins = np.logspace(5.6, 9.5, 40)   #Array of masses for each bin

    N_Trees_M = np.load('./All_Merger_Trees/N_Trees_Total.npy')     #Total N_trees in library by mass bin 
    deltas = np.load('./21cm/Remade_Bins_64_400.npy')               #Overdensity bins
    drawable = np.zeros((len(mass_bins), 4000))                     #Now initialize the 2D array of available merger trees (more massive half of bins)
    for im in range(0, len(mass_bins)):                 #Now loop through all mass bins with M > 10^8 M_sun 
        N_total = N_Trees_M[im]             #The total number of library trees for this mass bin
        drawable[im][N_total:] = 2.         #Make it so that trees which don't exist == 2
    print(drawable)

    for i in range(len(deltas)-1, -1, -1):              #Now loop through all overdensity bins & sample N_trees by halo mass bin
        Num_Halos = np.load('./HMFs/Num_Halos_' + str(i) + special + '.npy')    #The HMF for this overdensity to be Poisson sampled
        array_path = './z_arrays/Trees_' + str(i) + special                     #Path to save all z_arrays
        print('Current delta ID & value: ', i, round(deltas[i], 2))
        try:
            os.mkdir(array_path)    #Create a new directory for new runs
        except OSError:             #Skip if it's already been created
            print(OSError)
        all_tree_IDs = []                                   #Initialize list of all drawn merger tree IDs
        poisson = np.random.poisson(Num_Halos).astype(int)  #Get a random poisson number of trees per halo mass bin
        np.save(array_path + '/Num_Trees.npy', poisson)     #And save it
        for im in range(0, len(mass_bins)):                 #Now loop through halo mass bins to randomly Poisson sample that many trees for each
            poisson_im = min(poisson[im], 100)      #The number of physical trees used for this mass bin (max 100 allowed per bin)
            tree_IDs = np.array([]).astype(int)     #Initialize array of ID's drawn for this bin
            if im < 24:                             #For low mass halos, allow duplicates of 4k trees in library
                tree_IDs = np.array(random.sample(range(N_Trees_M[im]), poisson_im)).astype(int)    #Get a random sample of trees for this mass bin from tree library
            elif poisson_im > 0:                    #For more massive halos, do not allow duplications
                library = drawable[im]              #Isolate the availability of library merger trees for this mass bin
                arange = np.arange(len(library))    #Initialize array of tree IDs for this bin
                available = arange[library < 0.5]   #Limit to the indicies which have yet to be drawn (=0)
                try:
                  tree_IDs = np.array(np.random.choice(available, size=poisson_im, replace=False)).astype(int)   #Randomly sample poisson_im unused trees for this mass bin
                except:
                  print('Unable to not repeat for mass bin -- ', im, poisson_im, N_Trees_M[im])
                library[tree_IDs] = 1.                                              #Update list with trees that were just drawn to prevent them from being drawn 
            all_tree_IDs.append(tree_IDs)                                           #Add these IDs to overall list of IDs
        np.save(array_path + '/Tree_IDs.npy', all_tree_IDs, allow_pickle=True)      #And the IDs of the merger trees used

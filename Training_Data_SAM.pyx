import numpy as np
import scipy.integrate, os, time, shutil, random, sys
cimport numpy as np
cimport cython
 
def SFR_ODE(double M_halo, int ID_prog1, int ID_prog2, np.ndarray[np.float64_t,ndim=1] popII, np.ndarray[np.float64_t,ndim=1] m_gas, np.ndarray[np.float64_t,ndim=1] popIII_on, double delta_t, double tau_ff, double z_pres, double t_present, double t_popIII_on, double t_delay, double eta_SN, int ID, double fII):
    cdef double t_test, popII1, popII2, M0, M1, m_cg, m_acc, coeff, gas_mass, popII_new, popII_mass, ejected_mass
    cdef np.ndarray[np.float64_t,ndim=1] output

    output = np.array([])                       #Initialize output array
    if popIII_on[ID] > 1e-8:                    #Halo must have popIII already
        t_test = t_present - t_popIII_on        #Difference in time between present and time popIII SF turned on
        if t_test >= t_delay:                                   #Assign popII stars after t_delay
            popII1, popII2 = popII[ID_prog1], popII[ID_prog2]   #PopII stellar mass in both progenitors
            M0 = max(m_gas[ID_prog1], m_gas[ID_prog2])          #Finding initial gas mass
            M1 = min(m_gas[ID_prog1], m_gas[ID_prog2])          #Gas mass that will "accrete" over delta_t
            m_cg, coeff = M1/delta_t, fII * (1.+eta_SN)         #m_c,g from Furlanetto, delta_m_gas/delta_t, and variable I defined for the epsilon_ff*(1+eta) 
            m_acc = m_cg * tau_ff                               #m_acc from Furlanetto, accretion rate * ff time
            if popII1 > 1e-8 and popII2 > 1e-8 or M1 < 1e-8:                #If both halos have formed popII in the past, or smaller mass = 0 (sometimes?) grow them independently
                gas_mass = (M0 + M1) * np.exp(-(coeff*delta_t)/tau_ff)      #Adding analytic gas mass solutions of the two progenitors, and solutions below for SF to get new M*
                popII_new = ((m_gas[ID_prog1])/(1.+eta_SN)) * (1.-np.exp(-(coeff*delta_t)/tau_ff)) + ((m_gas[ID_prog2])/(1.+eta_SN)) * (1.-np.exp(-(coeff*delta_t)/tau_ff))
            else:
                gas_mass = (1./coeff)*((coeff*M0 - m_acc)*np.exp(-(coeff*delta_t)/tau_ff) + m_acc)    #Analytic gas mass at time t -- solution to EQ 4 in Furlanetto
                popII_new = (m_acc*fII/coeff)*((delta_t/tau_ff) + (M0/m_acc - 1./coeff)*(1.-np.exp(-coeff*delta_t/tau_ff)))
            popII_mass = popII1 + popII2 + popII_new                        #Add progenitor popII stellar masses to newly formed stars
            ejected_mass = M0 + M1 - gas_mass - popII_new                   #Gas mass lost from present halo is value now missing from final gas (M0 + M1) - (new stellar mass)
            output = np.array([popII_mass, gas_mass, ejected_mass])
    return(output)

# ------ Above is the code for determining PopII star formation, below is the full SAM framework for generating neural network training data --------------------------------

def LW_combined(int JLW_i, int vbc_i, int delta, str special=''):
    cdef int checking, im, iN, z_start, iz, N_halos, ii, ID, ID_prog1, ID_prog2, poisson_im
    cdef str array_path, save_path
    cdef list all_trees, bin_trees, popIII_m_all, popII_m_all, m_gas_all, m_lost_all, polluted_all, popIII_all, popIII_on_all, popII_all, popIII_m_bin, popII_m_bin, m_gas_bin, m_lost_bin, polluted_bin, popIII_bin, popIII_on_bin, popII_bin, bin_lengths
    cdef double V_com, fII, fIII, z_pres, z_prev, time_z_arrays, H_0, h, Omega_b, Omega_m, Omega_L, E_SN, epsilon_SN, R_C_cgs, f_sigma, t_delay, start, J, M_a, M_H2, M_crit_0, t_pres, delta_t, tau_ff, m_gas_z, M_crit, M_halo, M_DM, M_baryon, M_stellar, M_ejected, M_gas_max, star_m, z_calc, Star_prog1, Star_prog2, popIIIprog1, popIIIprog2, t_popIII_on, eta_SN, popIII_sfr_sum, popII_sfr_sum, popIII_sfr, popII_sfr, end
    cdef np.ndarray[np.int_t,ndim=1] poisson, all_ID, polluted, ID_present, last_ID, all_desc, ID_prog, prog_ff
    cdef np.ndarray[np.float64_t,ndim=1] z, z_ff, t, tau, mass_bins, sigmas, D, all_z, J_z, M_crit_z, J_0, SFRD_II, SFRD_III, MMH, m_gas, m_lost, popIII, popIII_on, popII, popIII_m, popII_m, scatter, t_reroll, all_m, output, popIII_sfr_bar, popII_sfr_bar, M_star
    cdef np.ndarray[np.float64_t,ndim=2] all_halos, pres_array, prev_array, popIII_sfr_bar_all, popII_sfr_bar_all, last_halos

    start = time.time()                         #Immedately record start time to determine runtime
    z = np.linspace(15., 60., 901)              #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))          #Hubble time array
    z_ff = np.load('./z_ff_15.npy')             #Freefall time ago Redshift array
    tau = np.load('./tau_15.npy')               #Freefall time used in SFR_II calculation
    mass_bins = np.logspace(5.6, 9.5, 40)       #Array of masses for each bin
    V_com, fII, fIII = 27., 0.0025, 0.0001      #Comoving volume & PopII/PopIII SFEs
    H_0, Omega_b, Omega_m, t_delay, E_SN, epsilon_SN = 67.0, 0.049, 0.32, 1e7, 1e51, 1.6e-3  #Various constants
    h, Omega_L = H_0/100., 1.-Omega_m           #Various needed constants, little h, curvature density param
    R_C_cgs = (6.8e48/E_SN) * 1.989e-33         #SN energy per M_sun in units g^-1
    eta_SN = 0.0                                #SNe gas ejection efficiency

    array_path = './z_arrays/Trees_' + str(delta)                   #Path with z_arrays & tree ID's/info
    save_path = './NN_Training/Deltas/Delta_' + str(delta) + '/'    #Path where results will be saved for this run
    checking = 0        #Initialize check value to make sure we haven't ran this SAM yet
    try:                #Try loading in the SF data for this set of input parameters
        data = np.load(save_path + '/SF_' + str(vbc_i) + '_' + str(JLW_i) + special + '.npy')
        checking = 1    #If it loads in, update checking value
    except:             #If it doesn't load, checking is not updated
        print(checking) #Continue regardless of outcome
    else:
        if checking == 1:           #Now, check if checking is 0 or 1 -- If it's 1, then SF array loaded and we don't need to run this SAM
            print('Already ran')    #So leave the simulation
            sys.exit()
    poisson = np.load(array_path + '/Num_Trees.npy')[:len(mass_bins)].astype(int)   #Load in the number of trees in each mass bin
    all_lengths = np.load(array_path + '/Tree_Lengths.npy', allow_pickle=True)      #And tree lengths
    J_z = np.load('./J_LW_0/Training_Js/J_' + str(JLW_i) + '.npy')                  #Load in LW background intenstiy
    M_crit_z = np.load('./M_Crits/M_crit_' + str(vbc_i) + '_' + str(JLW_i) + '_2.npy')  #And corresponding M_crit

    SFRD_II, SFRD_III = np.zeros(len(z)), np.zeros(len(z))      #Initialize SFRD arrays for both stellar poulations
    M_star, SF_II = np.zeros(len(z)), np.zeros(len(z))          #Also initialize emulated SF value arrays
    popIII_sfr_bar_all = np.zeros((len(mass_bins), len(z)))     #SFR averaged within a mass bin
    popII_sfr_bar_all = np.zeros((len(mass_bins), len(z)))      #Initialize all 'grand' lists to avoid reading/writing/saving/loading arrays 
    popIII_m_all, popII_m_all, m_gas_all, m_lost_all, polluted_all, popIII_all, popIII_on_all, popII_all = [],[],[],[],[],[],[],[]
    for im in range(0, len(mass_bins)):                         #Also initialize all bin lists to append onto grand lists
        popIII_m_bin, popII_m_bin, m_gas_bin, m_lost_bin, polluted_bin, popIII_bin, popIII_on_bin, popII_bin = [],[],[],[],[],[],[],[]
        poisson_im = min(100, poisson[im])                      #Set maximum number of trees per mass bin = 100
        for iN in range(0, poisson_im):                         #Loop through all trees in this mass bin
            N_halos = all_lengths[im][iN]                       #The total number of halos in the tree
            popIII_m_bin.append(np.zeros(len(z)))               #Mass of popIII stars at each z step
            popII_m_bin.append(np.zeros(len(z)))                #And mass of popII at each z
            m_gas_bin.append(np.zeros(N_halos))                 #Mass of the gas in a halo
            m_lost_bin.append(np.zeros(N_halos))                #Mass of the gas ejected by SNe at each z step
            polluted_bin.append(np.zeros(N_halos).astype(int))  #Polluted -- had stars at previous step as well
            popIII_bin.append(np.zeros(N_halos))                #Number of halos with Pop III stars at each redshift step
            popIII_on_bin.append(np.zeros(N_halos))             #Redshift step when a halo first acquired PopIII stars
            popII_bin.append(np.zeros(N_halos))                 #Mass of PopII stars in a halo
        popIII_m_all.append(popIII_m_bin)
        popII_m_all.append(popII_m_bin)
        m_gas_all.append(m_gas_bin)
        m_lost_all.append(m_lost_bin)
        polluted_all.append(polluted_bin)
        popIII_all.append(popIII_bin)
        popIII_on_all.append(popIII_on_bin)
        popII_all.append(popII_bin)

    z_start = np.load(array_path + '/z_start.npy')  #Load in z index where all_halos runs out
    iz = z_start-1                                  #Find index of that redshift
    z_pres = round(z[iz], 2)                        #Start model at the highest resdhift with halos
    z_arrays = np.load(array_path + '/' + str(z_pres) + '_Arrays.npy', allow_pickle=True)       #Load in current z_arrays
    J, M_crit, t_pres, delta_t, tau_ff = J_z[iz], M_crit_z[iz], t[iz], t[iz]-t[iz+1], tau[iz]   #Get relevant values for this z
    print(J, M_crit, z_pres)
    for im in range(0, len(mass_bins)):             #Loop through all halo mass bins for highest z
        poisson_im = min(100, poisson[im])          #Number of trees actually used in this mass bin
        for iN in range(0, poisson_im):             #Loop calculating stellar mass (among other things) for every tree
            m_gas = m_gas_all[im][iN]               #The current tree's gas mass for all halos
            m_lost = m_lost_all[im][iN]             #It's ejected mass for all halos
            polluted = polluted_all[im][iN]         #It's polluted status ""
            popIII = popIII_all[im][iN]             #PopIII mass ""
            popIII_on = popIII_on_all[im][iN]       #Time of PopIII turn on ""
            popIII_m = popIII_m_all[im][iN]         #PopIII mass at each z step for this tree
            all_halos = z_arrays[im][iN]                        #Current halos, redshift, masses, and descendent halos
            ID_present = all_halos[:,0].astype(int)             #ID's of each halo
            for ii in range(0, len(ID_present)):                #Loop through current halos for this tree
                ID, M_halo = ID_present[ii], all_halos[ii,2]    #Present halo ID & mass - Next line is halo escape velocity
                M_DM = ((Omega_m - Omega_b)/Omega_m) * M_halo   #DM mass of the halo
                M_baryon = M_halo - M_DM                        #Baryonic mass of the halo
                M_gas_max = M_baryon                            #Maximum amount of gas, which on first z step is M_baryon
                if M_gas_max < 0.0:                             #Cannot have negative gas mass...
                    M_gas_max = 0.0                             #So set it to zero
                m_gas[ID] = M_gas_max
                star_m, z_calc = 0.0, float(z_pres) + 0.05      #Reset PopIII stellar mass for this halo, and previous redshift step
                if M_halo >= M_crit and polluted[ID] < 1e-8:    #Assigning PopIII stars to a halo as soon as it's > M_crit
                    star_m = 200.0 * round(fIII/0.0001, 1)      #Stellar mass to be introduced
                    if star_m > M_gas_max:                      #Can't make more stars than you have gas
                        star_m = M_gas_max
                    popIII[ID], popIII_on[ID] = star_m, z_pres      #Assign stars and record redshift of first PopIII
                    m_gas[ID] = m_gas[ID] - star_m                  #Subtract stellar mass from gas mass
    #SFR/SFRD CODE --------------------------------------------------------------------------------------------------------
    for im in range(0, len(mass_bins)):     #After all trees have calculated popIII & II masses, loop through again to determine sfr
        if poisson[im] < 0.5:               #Skip the mass bin if there are no trees
            continue
        popIII_sfr_bar, popII_sfr_bar = popIII_sfr_bar_all[im], popII_sfr_bar_all[im]   #Load the SFR avg arrays for current mass bin
        popIII_sfr_sum, popII_sfr_sum, poisson_im = 0., 0., min(100, poisson[im])       #Reset the two sum values for this mass bin & N_trees(M)
        for iN in range(0, poisson_im):                                                 #Loop through the number of trees for each mass bin 
            popIII_m, popII_m = popIII_m_all[im][iN], popII_m_all[im][iN]   #Isolate the stellar masses of this tree
            popIII_sfr_sum += abs(popIII_m[iz+1] - popIII_m[iz]) / delta_t  #Change in stellar mass divided by change in time
            popII_sfr_sum += abs(popII_m[iz+1] - popII_m[iz]) / delta_t                             #For both stellar populations
        popIII_sfr_bar[iz], popII_sfr_bar[iz] = popIII_sfr_sum/poisson_im, popII_sfr_sum/poisson_im #Average over number of trees
    for im in range(0, len(mass_bins)):                                                             #Now loop through again to determine SFRD
        SFRD_II[iz] += (poisson[im]/V_com)*popII_sfr_bar_all[im][iz]        #Weight SFR by the # of halos in that mass bin and divide by volume
        SFRD_III[iz] += (poisson[im]/V_com)*popIII_sfr_bar_all[im][iz]      #For both stellar populations
    print(SFRD_III[iz], SFRD_II[iz])
    M_star[iz] = (SFRD_III[iz]*V_com*delta_t/200.) + M_star[iz+1]
    SF_II[iz] = (SFRD_II[iz]*V_com*delta_t) + SF_II[iz+1]
    #Above is the SF code for the very first halos, ensuring if any halos are > M_crit they have SF, below begins the time evolution ------------------------
    for iz in range(z_start-2, -1, -1):                             #Start at high z, early time, and step forward through time
        z_pres, z_prev = round(z[iz], 2), round(z[iz+1], 2)         #Current and previous resdhift z values
        z_arrays = np.load(array_path + '/' + str(z_pres) + '_Arrays.npy', allow_pickle=True)
        z_arrays_0 = np.load(array_path + '/' + str(z_prev) + '_Arrays.npy', allow_pickle=True)
        J, M_crit = J_z[iz], M_crit_z[iz]                           #Current LW background and M_crit
        t_pres, delta_t, tau_ff = t[iz], t[iz]-t[iz+1], tau[iz]     #Current Hubble time, change since last step, and freefall time ago
        print(J, M_crit, z_pres)
        for im in range(0, len(mass_bins)):
            poisson_im = min(100, poisson[im])                      #Loop through all trees in each mass bin
            for iN in range(0, poisson_im):                         #Main loop, calculating stellar mass (among other things) for every tree
                #Isolate all characteristics of this particular merger tree ---------------------------------------------------------------
                m_gas, m_lost, polluted, popIII_on = m_gas_all[im][iN], m_lost_all[im][iN], polluted_all[im][iN], popIII_on_all[im][iN]
                popIII, popII, popIII_m, popII_m = popIII_all[im][iN], popII_all[im][iN], popIII_m_all[im][iN], popII_m_all[im][iN]
                all_halos, last_halos = z_arrays[im][iN], z_arrays_0[im][iN]
                ID_present, last_ID =  all_halos[:,0].astype(int), last_halos[:,0].astype(int)
                all_z, all_m, all_desc = last_halos[:,1], last_halos[:,2], last_halos[:,3].astype(int)
                for ii in range(0, len(ID_present)):                                #Loop through current halos for this tree
                    ID, M_halo = ID_present[ii], all_halos[ii, 2]                   #Present halo mass
                    ID_prog = last_ID[np.abs(all_desc - float(ID)) < 1e-8]          #ID of the progenitors
                    M_DM = ((Omega_m - Omega_b)/Omega_m) * M_halo                   #DM mass in the halo
                    M_baryon = M_halo - M_DM                                        #M_halo = M_baryon + M_DM
                    M_stellar = np.sum(popIII[ID_prog]) + np.sum(popII[ID_prog])    #Calculate stellar mass within immediate progenitors
                    M_ejected = np.sum(m_lost[ID_prog])                             #And the ejected gas mass from progenitors
                    M_gas_max = M_baryon - M_stellar - M_ejected                    #Maximum amount of gas = M_halo - M_DM - M_stellar - M_ejected
                    if M_gas_max < 0.0:                                             #Cannot have negative gas mass...
                        M_gas_max = 0.0                                             #So set it to zero
                    m_gas[ID], star_m, z_calc = M_gas_max, 0., float(z_pres)+0.05                   #Reset new PopIII mass and get previous redshift
                    polluted[ID] = (np.sum(polluted[ID_prog]) + np.sum(popIII[ID_prog]>1e-8)) > 0   #If either progenitor was already polluted or has stellar mass, current halo is polluted
                    # - PopIII Star Formation --------------------------------------------------------------------------------
                    if M_halo >= M_crit and polluted[ID] < 1e-8:                    #Assigning PopIII stars to a halo as soon as it's > M_crit
                        star_m = 200.0 * round(fIII/0.0001, 1)                      #Stellar mass to introduce
                        if star_m > M_gas_max:                                      #Can't make more stars than you have gas
                            star_m = M_gas_max
                        popIII[ID], popIII_on[ID], m_gas[ID] = star_m, z_pres, m_gas[ID] - star_m   #Subtract stellar mass from gas mass
                    # - PopII Star Formation ---------------------------------------------------------------------------------
                    if len(ID_prog) == 1:                           #If the halo has only one progenitor...
                        popIII[ID] += popIII[ID_prog[0]]            #Make sure to add its PopIII stellar mass to current halo's total (regardless of star formation this z step)
                        if popIII_on[ID_prog[0]] > z[0]:            #Also only keep the PopIII turn on time of the halo that turned on first
                            popIII_on[ID] = popIII_on[ID_prog[0]]
                        t_popIII_on = 0.93e9 * (((1.+popIII_on[ID])/7.)**(-1.5))        #Hubble time that popIII turned on, calculate PopII star formation 
                        output = SFR_ODE(M_halo, ID_prog[0], 0, popII, m_gas, popIII_on, delta_t, tau_ff, z_pres, t_pres, t_popIII_on, t_delay, eta_SN, ID, fII)  #In the event of only 1 progenitor halo
                        if len(output) > 2:
                            popII[ID], m_gas[ID], m_lost[ID] = output[0], output[1], output[2]  #Record star formation if there is any
                    elif len(ID_prog) > 1.5:
                        ID_prog1, ID_prog2 = ID_prog[0], ID_prog[1]                     #Progenitor halo ID's if there are multiple
                        Star_prog1, Star_prog2 = popIII[ID_prog1], popIII[ID_prog2]     #Their stellar masses
                        popIII[ID] += Star_prog1 + Star_prog2                               #Adding stellar mass of the two progenitors together
                        popIIIprog1, popIIIprog2 = popIII_on[ID_prog1], popIII_on[ID_prog2] #Getting popIII_on progenitors to carry redshift value to next step
                        if popIIIprog1 > z[0] or popIIIprog2 > z[0]:                        #If either progenitor is polluted
                            popIII_on[ID] = max([popIIIprog1, popIIIprog2])             #Whichever redshift is higher is the assigned turn-on time
                        t_popIII_on = 0.93e9 * (((1.+popIII_on[ID])/7.)**(-1.5))        #Time that popIII turned on
                        output = SFR_ODE(M_halo, ID_prog1, ID_prog2, popII, m_gas, popIII_on, delta_t, tau_ff, z_pres, t_pres, t_popIII_on, t_delay, eta_SN, ID, fII)
                        if len(output) > 2:
                            popII[ID], m_gas[ID], m_lost[ID] = output[0], output[1], output[2]  #Record star formation if there is any
                popIII_m[iz] += np.sum(popIII[ID_present])                                      #These last lines record total M_star at each z step
                popII_m[iz] += np.sum(popII[ID_present])

        # - SFR CODE --------------------------------------------------------------------------------------------------------
        for im in range(0, len(mass_bins)):     #After all trees have calculated popIII & II masses, loop through again to determine sfr
            if poisson[im] < 0.5:               #Skip the mass bin if there are no trees
                continue
            popIII_sfr_bar, popII_sfr_bar = popIII_sfr_bar_all[im], popII_sfr_bar_all[im]   #Load the SFR avg arrays for current mass bin
            popIII_sfr_sum, popII_sfr_sum, poisson_im = 0., 0., min(100, poisson[im])       #Reset the two sum values for this mass bin & N_trees(M)
            for iN in range(0, poisson_im):                                                 #Loop through the number of trees for each mass bin 
                popIII_m, popII_m = popIII_m_all[im][iN], popII_m_all[im][iN]               #Isolate the stellar masses of this tree
                popIII_sfr_sum += abs(popIII_m[iz+1] - popIII_m[iz]) / delta_t              #Change in mass divided by change in time
                popII_sfr_sum += abs(popII_m[iz+1] - popII_m[iz]) / delta_t                 #Sum of star formation across different trees
            popIII_sfr_bar[iz], popII_sfr_bar[iz] = popIII_sfr_sum/poisson_im, popII_sfr_sum/poisson_im
        # - SFRD CODE -------------------------------------------------------------------------------------------------------
        for im in range(0, len(mass_bins)):                                     #After all SFRs have been calculated, loop through again to determine SFRD
            SFRD_II[iz] += (poisson[im]/V_com)*popII_sfr_bar_all[im][iz]        #Weight SFR by the # of halos in that mass bin and divide by volume
            SFRD_III[iz] += (poisson[im]/V_com)*popIII_sfr_bar_all[im][iz]      #For both stellar populations
        print(SFRD_III[iz], SFRD_II[iz])
        M_star[iz] = (SFRD_III[iz]*V_com*delta_t/200.) + M_star[iz+1]
        SF_II[iz] = (SFRD_II[iz]*V_com*delta_t) + SF_II[iz+1]

#    np.save(save_path + '/SF_' + str(vbc_i) + '_' + str(JLW_i) + special + '.npy', np.row_stack((SF_II, M_star))) #Finally, save star formation histories
    end = time.time() - start
    print('Run Time (min): ', round(end/60.,2))
    np.save('./NN_Training/Deltas/Runtimes/Run_' + str(delta) + '_' + str(vbc_i) + '_' + str(JLW_i) + '.npy', end)

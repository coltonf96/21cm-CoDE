import numpy as np
from scipy import special, interpolate
import scipy.integrate, os, time, shutil, random
cimport numpy as np
cimport cython

def SFR_ODE(int ID_prog1, int ID_prog2, int ID, np.ndarray[np.float64_t,ndim=1] popII, np.ndarray[np.float64_t,ndim=1] m_gas, np.ndarray[np.float64_t,ndim=1] popIII_on, np.ndarray[np.float64_t,ndim=1] vals):
    cdef int check
    cdef double t_test, popII1, popII2, M0, M1, m_cg, m_acc, coeff, gas_mass, popII_new, popII_mass, ejected_mass, T_vir, M_halo, delta_t, tau_ff, z_pres, t_pres, t_popIII_on, t_delay, eta_SN, fII, Omega_z_m, del_c, h, Omega_m, t_quies, T_steady, t_quies_halo
    cdef np.ndarray[np.float64_t,ndim=1] output

    M_halo, delta_t, tau_ff, z_pres, t_pres, t_popIII_on, t_delay = vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]
    eta_SN, t_quies_halo, fII, Omega_z_m, del_c, h, Omega_m, T_steady = vals[7], vals[8], vals[9], vals[10], vals[11], vals[12], vals[13], vals[14]
    output, check = np.array([]), 0                             #Gather parameters, and initialize output array + check for which PopII SF is happening
    if popIII_on[ID] > 1e-8:                                    #Halo must have popIII already
        t_test = t_pres - t_popIII_on                           #Difference in time between present and time popIII SF turned on
        if t_test >= t_delay:                                   #Only assign popII stars after t_delay or t_queiscent elapses -- Either bursty or steady
            popII1, popII2 = popII[ID_prog1], popII[ID_prog2]   #PopII stellar mass in both progenitors
            popII_new = 0.                                      #Reset new PopII mass in case none is made for this halo
            M0 = max(m_gas[ID_prog1], m_gas[ID_prog2])          #Finding initial gas mass
            M1 = min(m_gas[ID_prog1], m_gas[ID_prog2])          #Gas mass that will "accrete" over delta_t -- Calculate halo virial temperature [K]
            T_vir = 1.98e4 * (M_halo/(1e8/h))**(2./3.) * ((Omega_m*del_c)/(18.*(np.pi**2.)*Omega_z_m))**(1./3.) * ((1.+z_pres)/10.)   #EQ 26 of Barkana & Loeb 2001 -- Assuming mu=0.6
            # -- Bursty phase of PopII SF ---------------------------------------------
            if M_halo < 4e6 and T_vir < T_steady and t_pres >= t_quies_halo and fII < 1e-15:                #For halos that are below Renaissance mass range, use AEOS distribution
                popII_new = min(10.**(0.49794*np.log10(M_halo)+0.32442), m_gas[ID])
                t_quies = 2.55e4 * (M_halo)**-0.394
                gas_mass = m_gas[ID]
            elif M_halo >= 4e6 and T_vir < T_steady and t_pres >= t_quies_halo and fII < 1e-15:             #Alternatively, for halos within Ren. mass range, use distribution from Hazlett+ 2025
                popII_new = min(10.**(1.09180915*np.log10(M_halo) + 0.57809887*np.log10(1.+z_pres) - 4.66526393), m_gas[ID])
                t_quies = 2.55e4 * (M_halo)**-0.394
                gas_mass = m_gas[ID]
            # -- Steady phase of PopII SF ---------------------------------------------
            elif T_vir > T_steady or fII > 1e-15:                               #Whenever halo reaches T_vir, it is assigned an SFE, so either condition satisfies steady state
                if fII < 1e-15:                                                 #If this is the first time we're reaching steady state...
                    fII = 0.0139
                m_cg, coeff = M1/delta_t, fII * (1.+eta_SN)                     #m_c,g from Furlanetto, delta_m_gas/delta_t, and variable I defined for the epsilon_ff*(1+eta) 
                m_acc = m_cg * tau_ff                                           #m_acc from Furlanetto, accretion rate * ff time
                if popII1 > 1e-8 and popII2 > 1e-8 or M1 < 1e-8:                #If both halos have formed popII in the past, or smaller mass = 0 (sometimes?) grow them independently
                    gas_mass = (M0 + M1) * np.exp(-(coeff*delta_t)/tau_ff)      #Adding analytic gas mass solutions of the two progenitors, and solutions below for SF to get new M*
                    popII_new = ((m_gas[ID_prog1])/(1.+eta_SN)) * (1.-np.exp(-(coeff*delta_t)/tau_ff)) + ((m_gas[ID_prog2])/(1.+eta_SN)) * (1.-np.exp(-(coeff*delta_t)/tau_ff))
                else:
                    gas_mass = (1./coeff)*((coeff*M0 - m_acc)*np.exp(-(coeff*delta_t)/tau_ff) + m_acc)    #Analytic gas mass at time t -- solution to EQ 4 in Furlanetto
                    popII_new = (m_acc*fII/coeff)*((delta_t/tau_ff) + (M0/m_acc - 1./coeff)*(1.-np.exp(-coeff*delta_t/tau_ff)))
                check, t_quies = 1, 0.      #Update check to make sure M_ej is calculated correctly, make quiescent time = 0 as we should be in steady state for remainder
            else:
                popII_new, t_quies, gas_mass = 0., (t_quies_halo-t_pres)/1e6, m_gas[ID]     #If it's not in steady state and mid-quiescent phase, assign values
            popII_mass, ejected_mass = popII1 + popII2 + popII_new, 0.                      #Add progenitor popII stellar masses to newly formed stars & initialize ejected gas mass
            if check == 1:
                ejected_mass = M0 + M1 - gas_mass - popII_new                               #Gas mass lost from present halo is value now missing from final gas (M0 + M1) - (new stellar mass)
            output = np.array([popII_mass, gas_mass, ejected_mass, t_pres+t_quies*1e6, fII, check, popII_new])    #Note, t_quies_halo is in Myr, multiplying by 1e6 to add to t_Hubble(z)
    return(output)

 #ABOVE ARE ALL OF THE USER DEFINED FUNCTIONS THAT ARE INPUTS FOR THE ENTIRE SEMIANALYTIC MODEL BELOW___________________________________________________________________

def LW_combined(str special, int N_sample, int N_star=4, double star_m0=35.69, double t_delay=1e7):
    cdef int x_coord, y_coord, z_coord, delta_i, vbc_i, poisson_im, im, iN, iz, N_halos, z_start, ii, ID, ID_prog1, ID_prog2
    cdef str save_path, array_path
    cdef list popIII_m_all, popII_m_all, m_gas_all, m_lost_all, polluted_all, popIII_all, popIII_on_all, popII_all, popIII_m_bin, popII_m_bin, m_gas_bin, m_lost_bin, polluted_bin, popIII_bin, popIII_on_bin, popII_bin
    cdef double V_com, fIII, z_pres, z_prev, time_z_arrays, H_0, h, Omega_b, Omega_m, Omega_L, E_SN, epsilon_SN, R_C_cgs, f_sigma, start, J, M_a, M_H2, M_crit_0, Omega_z_m, del_c, t_pres, delta_t, tau_ff, m_gas_z, M_crit, M_halo, M_DM, M_baryon, M_stellar, M_ejected, M_gas_max, star_m, z_calc, Star_prog1, Star_prog2, popIIIprog1, popIIIprog2, t_popIII_on, eta_SN, popII_new, popIII_sfr_sum, popII_sfr_sum, popIII_sfr, popII_sfr, end
    cdef np.ndarray[np.int_t,ndim=1] all_delta_i, all_vbc_i, coords, poisson, polluted, ID_present, last_ID, all_desc, ID_prog
    cdef np.ndarray[np.int_t,ndim=2] all_coords
    cdef np.ndarray[np.float64_t,ndim=1] z, z_ff, t, tau, masses, sigmas, D, all_z, J_z, M_crit_z, J_0, SFRD_II, SFRD_III, MMH, m_gas, m_lost, popIII, popIII_on, popII, popIII_m, popII_m, t_quieses, fIIs, all_m, output, popIII_sfr_bar, popII_sfr_bar, M_star
    cdef np.ndarray[np.float64_t,ndim=2] all_J_z, all_M_crit, popIII_sfr_bar_all, popII_sfr_bar_all, all_halos, last_halos

    z = np.linspace(15., 60., 901)          #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))      #Hubble time array
    z_ff = np.load('../z_ff_15.npy')        #Freefall time ago Redshift array
    tau = np.load('../tau_15.npy')          #Freefall time used in SFR_II calculation
    masses = np.logspace(5.6, 9.5, 40)      #Array of halo masses for each bin
    V_com, fII, fIII = 27., 0.0025, 0.0001  #Comoving cell volume [Mpc^3] & PopII/PopIII SFEs
    vbcs = np.linspace(0., 3., 301)                 #All streaming velocity bins [sigma_vbc]
    deltas = np.load('../Delta_Bins_64_400.npy')    #All overdensity bins
    H_0, Omega_b, Omega_m, E_SN, epsilon_SN = 67., 0.049, 0.32, 1e51, 1.6e-3  
    h, Omega_L = H_0/100., 1.-Omega_m               #Various necessary constants
    N_side, eta_SN, T_vir_steady = 64, 0., 1.76e4   #SNe gas ejection efficiency & T_vir for steady PopII SF

    save_path = './Box_' + str(N_side) + special + '/Testing/'
    try:
        os.mkdir(save_path + 'SAM_Results')   #Make path to save results for this run (if needed)
    except OSError:
        print(OSError)

    all_coords = np.load(save_path + 'Random_Samples.npy').astype(int)
    all_delta_i = np.load(save_path + 'RS_delta_i.npy').astype(int)
    all_vbc_i = np.load(save_path + 'RS_vbc_i.npy').astype(int)     #Load in all sample data
    all_vbc = np.load(save_path + 'RS_vbc.npy')
    all_J_z = np.load(save_path + 'RS_J_z.npy')
    all_M_crit = np.load(save_path + 'RS_Mcrit_z.npy')
    coords = all_coords[N_sample]                                   #And get info for this particular cell
    x_coord, y_coord, z_coord = coords[0], coords[1], coords[2]     #Its coordinates
    delta_i, vbc_i = all_delta_i[N_sample], all_vbc_i[N_sample]     #Overdensity & vbc bins
    sigma_v, delta = vbcs[vbc_i], deltas[delta_i]                                   #The cell sigma_vbc and overdensity values
    J_z, M_crit_z, vbc = all_J_z[N_sample], all_M_crit[N_sample], all_vbc[N_sample] #Its J_LW(z), M_crit(z), and vbc value
    del all_coords, all_delta_i, all_vbc_i, all_vbc, all_J_z, all_M_crit            #Delete random sample data arrays for memory
    print('Cell Coordinates: ', coords)
    print('sigma_v & delta: ', sigma_v, delta)

    array_path = '../z_arrays/Trees_' + str(delta_i)                        #Path with z_arrays & tree ID's/info, and below is the particular run
    poisson = np.load(array_path + '/Num_Trees.npy').astype(int)                #Load in the number of trees in each mass bin
    all_tree_IDs = np.load(array_path + '/Tree_IDs.npy', allow_pickle=True)     #And the merger tree IDs
    all_lengths = np.load(array_path + '/Tree_Lengths.npy', allow_pickle=True)  #And tree lengths
    J_0 = np.load('../J_LW_0/J_0/J_0_' + str(round(sigma_v,2)) + '.npy')        #Array of LW = 0 M_Crit values for this sigma_vbc
    MMH = np.load(array_path + '/MMH.npy')                                  #And MMH(z) for this merger history
    print('Number of Trees', poisson)

    SFRD_II, SFRD_III = np.zeros(len(z)), np.zeros(len(z))      #Initialize SFRD arrays for both stellar poulations
    M_star, SF_II = np.zeros(len(z)), np.zeros(len(z))          #Also initialize emulated SF value arrays
    M_burst, M_steady = np.zeros(len(z)), np.zeros(len(z))      #Initialize PopII mass from bursts and from steady SF 
    popIII_sfr_bar_all = np.zeros((len(masses), len(z)))        #SFR averaged within a mass bin
    popII_sfr_bar_all = np.zeros((len(masses), len(z)))         #Initialize all 'grand' lists to avoid reading/writing/saving/loading arrays 
    popIII_m_all, popII_m_all, m_gas_all, m_lost_all, polluted_all, popIII_all, popIII_on_all, popII_all, t_quies_all, fII_all = [],[],[],[],[],[],[],[],[],[]
    for im in range(0, len(masses)):                            #Also initialize all bin lists to append onto grand lists
        popIII_m_bin, popII_m_bin, m_gas_bin, m_lost_bin, polluted_bin, popIII_bin, popIII_on_bin, popII_bin, t_quies_bin, fII_bin = [],[],[],[],[],[],[],[],[],[]
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
            t_quies_bin.append(np.zeros(N_halos))               #Quiescent periods between bursty PopII SF
            fII_bin.append(np.zeros(N_halos))                   #PopII SFEs of all halos
        popIII_m_all.append(popIII_m_bin)
        popII_m_all.append(popII_m_bin)
        m_gas_all.append(m_gas_bin)
        m_lost_all.append(m_lost_bin)
        polluted_all.append(polluted_bin)
        popIII_all.append(popIII_bin)
        popIII_on_all.append(popIII_on_bin)
        popII_all.append(popII_bin)
        t_quies_all.append(t_quies_bin)
        fII_all.append(fII_bin)

    z_start = np.load(array_path + '/z_start.npy')  #Load in z index where all_halos runs out
    iz = z_start-1                                  #Find the corresponding index in overall z array
    z_pres = round(z[iz], 2)                        #Start at the highest resdhift with halos
    z_arrays = np.load(array_path + '/' + str(z_pres) + '_Arrays.npy', allow_pickle=True)           #Load in that z array
    J, M_crit, t_present, delta_t, tau_ff = J_z[iz], M_crit_z[iz], t[iz], t[iz]-t[iz+1], tau[iz]    #Denote current parameter values
    Omega_z_m = (Omega_m*((1.+z_pres)**3.)) / (Omega_m*((1.+z_pres)**3.) + Omega_L)                 #EQ 23 from Barkana & Loeb 2001
    del_c = 18.*(np.pi**2.) + 82.*(Omega_z_m-1.) - 39.*((Omega_z_m-1.)**2.)                         #EQ 22 "" -- Both used in determining steady PopII SF
    print(z_pres, J, M_crit, MMH[iz])
    for im in range(0, len(masses)):                                                #Loop through all halo mass bins
        poisson_im = min(100, poisson[im])  #Number of trees used in this mass bin
        for iN in range(0, poisson_im):     #Loop calculating stellar mass (among other things) for every tree
            m_gas = m_gas_all[im][iN]       #The current tree's gas mass for all halos
            m_lost = m_lost_all[im][iN]             #It's ejected mass for all halos
            polluted = polluted_all[im][iN]         #It's polluted status ""
            popIII = popIII_all[im][iN]             #PopIII mass ""
            popIII_on = popIII_on_all[im][iN]       #Time of PopIII turn on ""
            popIII_m = popIII_m_all[im][iN]         #PopIII mass at each z step for this tree
            all_halos = z_arrays[im][iN]            #Current halos, redshift, masses, and descendent halos
            ID_present = all_halos[:,0].astype(int) #ID's of each halo
            for ii in range(0, len(ID_present)):                #Loop through current halos for this tree
                ID, M_halo = ID_present[ii], all_halos[ii,2]    #Present halo ID & mass - Next line is halo escape velocity
                M_DM = ((Omega_m - Omega_b)/Omega_m) * M_halo   #DM mass in the halo
                M_baryon = M_halo - M_DM                        #M_halo = M_baryon + M_DM
                M_gas_max = M_baryon                #Set maximum amount of gas = M_halo - M_DM - M_stellar - M_ejected
                if M_gas_max < 0.0:                 #Cannot have negative gas mass...
                    M_gas_max = 0.0                 #So set it to zero if it is
                m_gas[ID], star_m = M_gas_max, 0.   #Record gas mass for this halo & reset mass of popIII stars that form in it
                z_calc = float(z_pres) + 0.05                   #Getting previous redshift, even at the end of the all_halos array
                if M_halo >= M_crit and polluted[ID] < 1e-8:    #Assign PopIII stars as soon as M_halo > M_crit
                    star_m = star_m0 * float(N_star)            #Stellar mass to be introduced (four PopIII stars with M=35.69 M_sun)
                    if star_m > M_gas_max:                      #Limit stellar mass to max gas mass in the halo
                        star_m = M_gas_max
                    popIII[ID], popIII_on[ID] = star_m, z_pres  #Record PopIII mass of halo, record z that popIII turns on
                    m_gas[ID] = m_gas[ID] - star_m              #Subtract stellar mass from gas mass
    #SFR/SFRD CODE --------------------------------------------------------------------------------------------------------
    for im in range(0, len(masses)):        #After all trees have calculated popIII & II masses, loop through again to determine sfr
        if poisson[im] < 0.5:               #Skip the mass bin if there are no trees
            continue
        popIII_sfr_bar, popII_sfr_bar = popIII_sfr_bar_all[im], popII_sfr_bar_all[im]   #Load the SFR avg arrays for current mass bin
        popIII_sfr_sum, popII_sfr_sum, poisson_im = 0., 0., min(100, poisson[im])       #Reset the two sum values for this mass bin & N_trees(M)
        for iN in range(0, poisson_im):                                                 #Loop through the number of trees for each mass bin 
            popIII_m, popII_m = popIII_m_all[im][iN], popII_m_all[im][iN]   #Isolate the stellar masses of this tree
            popIII_sfr_sum += abs(popIII_m[iz+1] - popIII_m[iz]) / delta_t  #Change in stellar mass divided by change in time
            popII_sfr_sum += abs(popII_m[iz+1] - popII_m[iz]) / delta_t                             #For both stellar populations
        popIII_sfr_bar[iz], popII_sfr_bar[iz] = popIII_sfr_sum/poisson_im, popII_sfr_sum/poisson_im #Average over number of trees
    for im in range(0, len(masses)):                                                                #Now loop through again to determine SFRD
        SFRD_II[iz] += (poisson[im]/V_com)*popII_sfr_bar_all[im][iz]        #Weight SFR by the # of halos in that mass bin and divide by volume
        SFRD_III[iz] += (poisson[im]/V_com)*popIII_sfr_bar_all[im][iz]      #For both stellar populations
    M_star[iz] = (SFRD_III[iz]*V_com*delta_t/200.) + M_star[iz+1]
    SF_II[iz] = (SFRD_II[iz]*V_com*delta_t) + SF_II[iz+1]
    print('SF: ', M_star[iz], SF_II[iz])
    #Above is the code for checking that the very first halos in a merger tree are > M_crit, below begins the time evolution ------------------------
    for iz in range(z_start-2, -1, -1):                         #Start at high z, early time, and step forward through time
        z_pres, z_prev = round(z[iz], 2), round(z[iz+1], 2)     #Current resdhift z value & previous step
        z_arrays = np.load(array_path + '/' + str(z_pres) + '_Arrays.npy', allow_pickle=True)       #Load in present z_arrays
        z_arrays_0 = np.load(array_path + '/' + str(z_prev) + '_Arrays.npy', allow_pickle=True)     #And z_arrays of previous step
        J, M_crit = J_z[iz], M_crit_z[iz]                       #Current LW Background intensity & critical mass
        t_pres, delta_t, tau_ff = t[iz], t[iz]-t[iz+1], tau[iz] #Present Hubble time, change since last z step, and freefall time (used in popII SFR calculation)
        print(z_pres, J, M_crit, MMH[iz])
        Omega_z_m = (Omega_m*((1.+z_pres)**3.)) / (Omega_m*((1.+z_pres)**3.)+Omega_L)   #EQ 23 from Barkana & Loeb 2001
        del_c = 18.*(np.pi**2.) + 82.*(Omega_z_m-1.) - 39.*((Omega_z_m-1.)**2.)         #EQ 22 "" -- Both used in determining steady PopII SF
        for im in range(0, len(masses)):                #Loop through all halo mass bins for this z
            poisson_im = min(100, poisson[im])          #Total number of trees to loop through for this mass bin
            for iN in range(0, poisson_im):             #Now loop through to calculate stellar mass (among other things) for every tree
                #Isolate all characteristics of this particular merger tree ---------------------------------------------------------------
                m_gas, m_lost, polluted, popIII_on, t_quieses = m_gas_all[im][iN], m_lost_all[im][iN], polluted_all[im][iN], popIII_on_all[im][iN], t_quies_all[im][iN]
                popIII, popII, popIII_m, popII_m, fIIs = popIII_all[im][iN], popII_all[im][iN], popIII_m_all[im][iN], popII_m_all[im][iN], fII_all[im][iN]
                all_halos, last_halos = z_arrays[im][iN], z_arrays_0[im][iN]
                ID_present, last_ID =  all_halos[:,0].astype(int), last_halos[:,0].astype(int)
                all_z, all_m, all_desc = last_halos[:,1], last_halos[:,2], last_halos[:,3].astype(int)
                for ii in range(0, len(ID_present)):                                #Loop through current halos for this tree
                    ID, M_halo = ID_present[ii], all_halos[ii, 2]                   #Present halo mass
                    T_vir = 1.98e4 * (M_halo/(1e8/h))**(2./3.) * ((Omega_m*del_c)/(18.*(np.pi**2.)*Omega_z_m))**(1./3.) * ((1.+z_pres)/10.)
                    ID_prog = last_ID[np.abs(all_desc - float(ID)) < 1e-8]          #ID of the progenitors
                    M_DM = ((Omega_m - Omega_b)/Omega_m) * M_halo                   #DM mass in the halo
                    M_baryon = M_halo - M_DM                                        #M_halo = M_baryon + M_DM
                    M_stellar = 0.
                    if T_vir > T_vir_steady or fIIs[ID] > 1e-15:
                        M_stellar = np.sum(popIII[ID_prog]) + np.sum(popII[ID_prog])    #Calculate stellar mass within immediate progenitors
                    M_ejected = np.sum(m_lost[ID_prog])                             #And the ejected gas mass from progenitors
                    M_gas_max = M_baryon - M_stellar - M_ejected                    #Maximum amount of gas = M_halo - M_DM - M_stellar - M_ejected
                    if M_gas_max < 0.0:                                             #Cannot have negative gas mass...
                        M_gas_max = 0.0                                             #So set it to zero
                    m_gas[ID], star_m, z_calc = M_gas_max, 0., float(z_pres)+0.05                   #Reset new PopIII mass and get previous redshift
                    polluted[ID] = (np.sum(polluted[ID_prog]) + np.sum(popIII[ID_prog]>1e-8)) > 0   #If either progenitor was already polluted or has stellar mass, current halo is polluted
                    # - PopIII Star Formation --------------------------------------------------------------------------------
                    if M_halo >= M_crit and polluted[ID] < 1e-8:    #Assigning PopIII stars to a halo as soon as it's > M_crit
                        star_m = star_m0 * float(N_star)            #Stellar mass to be introduced 
                        if star_m > M_gas_max:                      #Can't make more stars than you have gas
                            star_m = M_gas_max
                        popIII[ID], popIII_on[ID], m_gas[ID] = star_m, z_pres, m_gas[ID] - star_m   #Subtract stellar mass from gas mass
                    # - PopII Star Formation ---------------------------------------------------------------------------------
                    if len(ID_prog) == 1:                           #If the halo has only one progenitor...
                        popIII[ID] += popIII[ID_prog[0]]            #Make sure to add its PopIII stellar mass to current halo's total (regardless of star formation this z step)
                        if popIII_on[ID_prog[0]] > z[0]:            #Also only keep the PopIII turn on time of the halo that turned on first
                            popIII_on[ID] = popIII_on[ID_prog[0]]
                            fIIs[ID] = fIIs[ID_prog[0]]             #Also get its PopII SFE if it happens to have one
                            t_quieses[ID] = t_quieses[ID_prog[0]]                   #And its quiescent time for PopII SF to restart if applicable
                        t_popIII_on = 0.93e9 * (((1.+popIII_on[ID])/7.)**(-1.5))                    #Hubble time that popIII turned on, calculate PopII star formation 
                        popII_params = np.array([M_halo, delta_t, tau_ff, z_pres, t_pres, t_popIII_on, t_delay, eta_SN, t_quieses[ID], fIIs[ID], Omega_z_m, del_c, h, Omega_m, T_vir_steady])
                        output = SFR_ODE(ID_prog[0], 0, ID, popII, m_gas, popIII_on, popII_params)  #In the event of only 1 progenitor halo
                        if len(output) > 3:
                            popII[ID], m_gas[ID], m_lost[ID], t_quieses[ID], fIIs[ID], check, popII_new = output[0], output[1], output[2], output[3], output[4], int(output[5]), output[6]  #Record SF if there is any
                            if check == 1:                  #If check = 1, then we used ODEs
                                M_steady[iz] += popII_new   #So add the new PopII mass to M_steady
                            else:                           #Otherwise, if check = 0, it's still in bursty phase
                                M_burst[iz] += popII_new    #So add new PopII mass to the bursty mass array
                    elif len(ID_prog) > 1.5:
                        ID_prog1, ID_prog2 = ID_prog[0], ID_prog[1]                         #Progenitor halo ID's if there are multiple
                        Star_prog1, Star_prog2 = popIII[ID_prog1], popIII[ID_prog2]         #Their stellar masses
                        popIII[ID] += Star_prog1 + Star_prog2                               #Adding stellar mass of the two progenitors together
                        popIIIprog1, popIIIprog2 = popIII_on[ID_prog1], popIII_on[ID_prog2] #Getting popIII_on progenitors to carry redshift value to next step
                        if popIIIprog1 > z[0] or popIIIprog2 > z[0]:                        #If either progenitor is polluted....
                            popIII_on[ID] = max([popIIIprog1, popIIIprog2])                 #Record whichever redshift is higher is the assigned turn-on time
                        if t_quieses[ID_prog1] > 1e-8 and t_quieses[ID_prog2] > 1e-8:       #If both progenitors happen have PopII SFEs and/or at least t_quies assigned to them already
                            if fIIs[ID_prog1] > 1e-8 and fIIs[ID_prog2] < 1e-8:             #Whenever one progenitor has begun steady SF but the other is still bursty...
                                fIIs[ID] = fIIs[ID_prog1]                                   #Only the one will have fII assigned to it, so carry that value through
                            elif fIIs[ID_prog1] > 1e-8 and fIIs[ID_prog2] > 1e-8:               #Otherwise, if both progenitors are in steady SF...
                                fIIs[ID] = np.random.choice([fIIs[ID_prog1], fIIs[ID_prog2]])   #Take a random SFE of the progs, as in Hazlett+ 2025...
                            t_quieses[ID] = min([t_quieses[ID_prog1], t_quieses[ID_prog2]])     #Then take minimum t_quiesc of the two halos
                        elif t_quieses[ID_prog1] > 1e-8 or t_quieses[ID_prog2] > 1e-8:          #Now if just one of them is nonzero
                            fIIs[ID] = max([fIIs[ID_prog1], fIIs[ID_prog2]])                    #Take the max SFE for this halo (which would be the nonzero one)
                            t_quieses[ID] = max([t_quieses[ID_prog1], t_quieses[ID_prog2]])     #Repeat for t_quiescent
                        t_popIII_on = 0.93e9 * (((1.+popIII_on[ID])/7.)**(-1.5))                #Time that popIII turned on
                        popII_params = np.array([M_halo, delta_t, tau_ff, z_pres, t_pres, t_popIII_on, t_delay, eta_SN, t_quieses[ID], fIIs[ID], Omega_z_m, del_c, h, Omega_m, T_vir_steady])
                        output = SFR_ODE(ID_prog1, ID_prog2, ID, popII, m_gas, popIII_on, popII_params)
                        if len(output) > 3:
                            popII[ID], m_gas[ID], m_lost[ID], t_quieses[ID], fIIs[ID], check, popII_new = output[0], output[1], output[2], output[3], output[4], int(output[5]), output[6]  #Record SF if there is any
                            if check == 1:                  #If check = 1, then we used ODEs
                                M_steady[iz] += popII_new   #So add the new PopII mass to M_steady
                            else:                           #Otherwise, if check = 0, it's still in bursty phase
                                M_burst[iz] += popII_new    #So add new PopII mass to the bursty mass array
                popIII_m[iz] += np.sum(popIII[ID_present])  #These last lines record total M_star at each z step
                popII_m[iz] += np.sum(popII[ID_present])

        # - SFR CODE --------------------------------------------------------------------------------------------------------
        for im in range(0, len(masses)):        #After all trees have calculated popIII & II masses, loop through again to determine sfr
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
        for im in range(0, len(masses)):                                    #After all SFRs have been calculated, loop through again to determine SFRD
            SFRD_II[iz] += (poisson[im]/V_com)*popII_sfr_bar_all[im][iz]    #Weight SFR by the # of halos in that mass bin and divide by volume
            SFRD_III[iz] += (poisson[im]/V_com)*popIII_sfr_bar_all[im][iz]  #For both stellar populations
        print(z_pres, SFRD_III[iz], SFRD_II[iz])
        M_star[iz] = (SFRD_III[iz]*V_com*delta_t/200.) + M_star[iz+1]       #Convert SFRD_III to M_star & add to cumulative value
        SF_II[iz] = (SFRD_II[iz]*V_com*delta_t) + SF_II[iz+1]               #Also calculate cumulative PopII M_star
        print('SF: ', M_star[iz], SF_II[iz])

        if iz % 10 == 0:    #Overwrite previous arrays every dz = 0.5
            np.save(save_path + 'SAM_Results/SF_' + str(x_coord) + '_' + str(y_coord) + '_' + str(z_coord) + '.npy', np.row_stack((M_burst, M_steady, M_star)))
#            np.save(save_path + 'SAM_Results/Mstar_II_' + str(x_coord) + '_' + str(y_coord) + '_' + str(z_coord) + '.npy', np.array(popII_all, dtype=object), allow_pickle=True)
#            np.save(save_path + 'SAM_Results/Mstar_III_' + str(x_coord) + '_' + str(y_coord) + '_' + str(z_coord) + '.npy', np.array(popIII_all, dtype=object), allow_pickle=True)
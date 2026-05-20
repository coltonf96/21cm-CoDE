import numpy as np
cimport numpy as np
cimport cython

def Get_Data(int popIII, int popII, int delta, str special='', double t_delay=1e7):   # MAKE SURE THAT YOU RUN COMBINE_SF_OUTPUTS.PY FIRST ------------
  cdef int i, iz, delay_iz
  cdef str suffix
  cdef double vbc, z_pres, int_M, int_J, t_delay_ago, z_delay_ago, N_delay, M_crit_ago, int_N_t
  cdef np.ndarray[np.float64_t,ndim=1] z, t, vbcs, M_a, J_z, M_crit_z, MMH, cM_burst, cM_steady, cM_star, new_row_b, new_row_s

  z = np.linspace(15., 60., 901)        #Redshift values
  t = (0.93e9)*(((1.+z)/7.)**(-1.5))    #Hubble times
  vbcs = np.linspace(0.0, 3.0, 301)     #Sigma_vbc bins
#  all_inputs = np.load('./Deltas/All_Inputs.npy', allow_pickle=True)  #All LW/vbc combos
  all_inputs = np.load('./Deltas/All_Inputs_0.npy', allow_pickle=True)
  M_a = 5.4e7*(((1.+z)/11.)**-1.5)

  if popIII == 1:                       #If we need to gather PopIII data...
    all_data = []                       #Initialize the list of all data values
    print('Getting PopIII Data')        #And load in the SAM training data 
    all_SF = np.load('./Deltas/Delta_' + str(delta) + '/All_SF_Output_Data' + special + '.npy', allow_pickle=True)
    for i in range(0, len(all_inputs)):                                       #Loop through all LW/vbc combos
      inputs, SF = all_inputs[i], all_SF[i]                                   #Current LW/vbc values & their star formation results
      suffix = str(round(inputs[0],2)) + '_' + str(int(inputs[1])) + '_2.npy'   #File name suffix for this combo
      J_z = np.load('../J_LW_0/Training_Js/J_' + str(int(inputs[1])) + '.npy')#Load in J_LW(z) array
      M_crit_z = np.load('../M_Crits/M_crit_' + suffix)                       #And M_crit(z) array
      if special == '_Mcrit':
#          print('Multiplying M_crit by 2.25')
          M_crit_z *= 2.25
          M_crit_z[M_crit_z > M_a] = M_a[M_crit_z > M_a]
      MMH = np.load('../z_arrays/Trees_' + str(delta) + '/MMH.npy')           #Also load in MMH(z) for this halo merger history
      cM_star, vbc = SF[5,:], round(vbcs[inputs[0]],2)              #Cumulative (smoothed) M_PopIII and vbc parameter in terms of sigma_vbc
      for iz in range(0, len(cM_star)-10):              #Now loop back through time to gather PopIII SF data
        if sum(cM_star[iz:]) < 1e-15:               #Whenever we reach redshifts with no SF, skip them
          break
        z_pres = round(z[iz], 2)              #Current redshift
        int_M, int_J = np.trapz(M_crit_z[iz+1:], z[iz+1:]), np.trapz(J_z[iz+1:], z[iz+1:])    #Integrate M_crit & J_LW up to previous timestep
        if cM_star[iz] < 1e-15 or int_M < 1e-15 or int_J < 1e-15 or J_z[iz] < 1e-100 or MMH[iz] < 1e-15:
          continue                            #If any input value that needs logarithm taken are zero, skip it
        new_row = np.array([z_pres, np.log10(cM_star[iz]), np.log10(int_M), np.log10(int_J), vbc, np.log10(J_z[iz]), np.log10(MMH[iz])])
        all_data.append(new_row)              #If there are no zeros in the new row, take log values
    all_data = np.array(all_data)             #Convert list into array
    print(all_data[0])
    print(all_data.shape)
    np.save('./Deltas/Delta_' + str(delta) + '/PopIII_Box_Data' + special + '.npy', all_data, allow_pickle=True)
      
  if popII == 1:                      #Now if we need to gather PopII data...
    all_data_b, all_data_s = [], []   #Initialize list of all data values for both bursty & steady SF
    print('Getting PopII Data')       #And load in SAM training data
    all_SF = np.load('./Deltas/Delta_' + str(delta) + '/All_SF_Output_Data' + special + '.npy', allow_pickle=True)
    for i in range(0, len(all_inputs)):                                       #Now loop through all LW/vbc combos
      inputs, SF = all_inputs[i], all_SF[i]                                   #Current LW/vbc values & their star formation results
      suffix = str(round(inputs[0],2)) + '_' + str(int(inputs[1])) + '_2.npy'   #File name suffix for this combo
      M_crit_z = np.load('../M_Crits/M_crit_' + suffix)                       #Critical mass for this run
      if special == '_Mcrit':
#          print('Multiplying M_crit by 2.25')
          M_crit_z *= 2.25
          M_crit_z[M_crit_z > M_a] = M_a[M_crit_z > M_a]
      MMH = np.load('../z_arrays/Trees_' + str(delta) + '/MMH.npy')           #Also load in MMH(z) for this halo merger history
      cM_burst, cM_steady, cM_star = SF[1,:], SF[3,:], SF[5,:]      #PopII bursty & steady SF, also PopIII M_Star (all cumul)
      for iz in range(len(z)-10, -1, -1):           #Now step FORWARD through time
        if cM_burst[iz+1] < 1e-15:                  #...until we get to PopII SF
            continue
        z_pres = round(z[iz],2)                     #Current redshift
        t_delay_ago = t[iz] - t_delay               #Find Hubble time t_delay ago
        delay_iz = np.argmin(np.abs(t-t_delay_ago)) #And the nearest iz index
        z_delay_ago = round(z[delay_iz],2)          #To get the redshift 10 Myr ago
        N_delay = sum(cM_star[delay_iz:])           #The # of PopIII SF events 10 Myr ago (must be >0 to continue)
        M_crit_ago = M_crit_z[delay_iz]             #The M_crit a delay time ago
        if N_delay < 1e-12 or M_crit_ago < 1e-12:   #If either of those are zero for some reason, then skip this z step
            continue                                #Otherwise integrate PopIII M_star up to t_delay ago
        int_N_t = np.abs(np.trapz(cM_star[delay_iz:], t[delay_iz:]))    #And collect new row of data
        new_row_b = np.array([z_pres, np.log10(cM_burst[iz]), np.log10(int_N_t), np.log10(M_crit_ago)])
        all_data_b.append(new_row_b)    #Append to overall list
        if cM_steady[iz] > 1e-15:       #Once steady SF starts, begin recording it's data as well
          new_row_s = np.array([z_pres, np.log10(cM_steady[iz]), np.log10(int_N_t), np.log10(M_crit_ago), np.log10(MMH[iz])])
          all_data_s.append(new_row_s)
    all_data_b = np.array(all_data_b)   #Convert lists into arrays
    all_data_s = np.array(all_data_s)
    print(all_data_b[0])
    print(all_data_b.shape)
    np.save('./Deltas/Delta_' + str(delta) + '/PopII_Box_Data' + special + '_Bursty.npy', all_data_b, allow_pickle=True)
    if len(all_data_s) > 0.5:
        print(all_data_s[0])
        print(all_data_s.shape)
        np.save('./Deltas/Delta_' + str(delta) + '/PopII_Box_Data' + special + '_Steady.npy', all_data_s, allow_pickle=True)

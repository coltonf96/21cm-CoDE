import numpy as np
cimport numpy as np
cimport cython

def Smooth(int delta, str special=''):
  cdef int run_no, i, iz, ii, iz_smooth
  cdef str suffix
  cdef double V_com, M_pres, dM, dM_dz
  cdef np.ndarray[np.float64_t,ndim=1] z, t, SF_II, M_star, SF_dots, M_dots, SF_smooth, M_smooth
  cdef np.ndarray[np.float64_t,ndim=2] SF

  z = np.linspace(15., 60., 901)            #Redshift array
  t = (0.93e9)*(((1.+z)/7.)**(-1.5))        #Hubble time array
  V_com = 3.0**3.0                          #Cell volume in Mpc^3
  all_inputs = np.load('./Deltas/All_Inputs.npy', allow_pickle=True)
  print(len(all_inputs))                    #Load in all LW/vbc combos

  for i in range(0, len(all_inputs)):             #Loop through all of the runs and smooth their PopIII
      suffix = str(round(all_inputs[i][0],2)) + '_' + str(int(all_inputs[i][1])) + special   #Suffix for this LW/vbc combo
      try:
        SF = np.load('./Deltas/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy', allow_pickle=True)  #Load in SF data
      except:
        try:
          SF = np.load('./Deltas/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy')  #Sometimes it decides not to be pickled
        except:
          print('No file for ', suffix, i)
          continue
      if len(SF) > 2.5:
        print('Already Smoothed', i)
        continue
      SF_II, SF_dots = SF[0,:], np.zeros(len(z))  #Isolate PopII SF data & initialize array to record SF(z) values > SF(z+dz)
      M_star, M_dots = SF[1,:], np.zeros(len(z))  #Repeat for PopIII as in original simulation
      for iz in range(0, len(z)-1):               #Step through time
        if M_star[iz] > M_star[iz+1] or iz == 0:  #If this M_star(z) is > than the previous M_star(z+dz)...
          M_dots[iz] = M_star[iz]                 #Record it in M_dots, otherwise it remains 0
        if SF_II[iz] > SF_II[iz+1] or iz == 0:    #Similarly, for PopII, if it's higher than z+dz, record
          SF_dots[iz] = SF_II[iz]
      #Above is getting M_dots for both stellar populations, below is smoothiing between the two ----------------------------
      M_smooth, SF_smooth = np.zeros(len(z)), np.zeros(len(z))  #Now initialize smoothed stellar mass array for both
      for iz in range(0, len(z)):                 #Step through time to populate M_smooth
        if M_dots[iz] > 1e-15:                    #Only when M_dots has a nonzero value...
          M_pres = M_dots[iz]                     #Record it and begin the smoothing loop
          for ii in range(1, len(z)-iz-1):        #Step through, at most, the rest of the z array 
            if M_dots[iz+ii] > 1e-15:             #Find the next z where M_dot(z) is nonzero...
              iz_smooth = ii                      #Record it & break out of the for loop
              break
            if sum(M_dots[iz+ii:]) < 1e-15:       #Whenever we reach the last dot...
              iz_smooth = 1                       #Use dz = 0.05 as the last iz_smooth value & break out of loop
              break
          dM = M_pres - M_dots[iz+iz_smooth]      #Now determine how much M_star changes from M_pres to the next nonzero value
          dM_dz = dM/iz_smooth                    #Divide by # of smoothing steps to get smoothing value for each z step
          for ii in range(0, iz_smooth):          #Now step back through z array to that next nonzero value
            M_smooth[iz+ii] = M_pres - ii*dM_dz   #And populate M_smooth with step_size*N_step
        #Above is PopIII 'if' statement for smoothing, below is PopII 'if' statement -----------------------------------------
        if SF_dots[iz] > 1e-15:                   #Repeat above 'if' statement for PopII
          M_pres = SF_dots[iz]
          for ii in range(1, len(z)-iz-1):        #Step through, at most, the rest of the z array 
            if SF_dots[iz+ii] > 1e-15:            #Find the next z where M_dot(z) is nonzero...
              iz_smooth = ii                      #Record it & break out of the for loop
              break
            if sum(SF_dots[iz+ii:]) < 1e-15:      #Whenever we reach the last dot...
              iz_smooth = 1                       #Use dz = 0.05 as the last iz_smooth value & break out of loop
              break
          dM = M_pres - SF_dots[iz+iz_smooth]     #Now determine how much M_star changes from M_pres to the next nonzero value
          dM_dz = dM/iz_smooth                    #Divide by # of smoothing steps to get smoothing value for each z step
          for ii in range(0, iz_smooth):          #Now step back through z array to that next nonzero value
            SF_smooth[iz+ii] = M_pres - ii*dM_dz  #And populate SF_smooth with step_size*N_step
      M_smooth = np.nan_to_num(M_smooth)          #Remove NaNs from smoothed SF array 
      SF_smooth = np.nan_to_num(SF_smooth)        #For both stellar populations -- Overwrite SF file with smoothed values
      np.save('./Deltas/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy', np.row_stack((SF_II, SF_smooth, M_star, M_smooth)))
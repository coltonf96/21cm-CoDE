import numpy as np
cimport numpy as np
cimport cython

def Smooth(int delta, str special='_Burst'):
  cdef int run_no, i, iz, ii, iz_smooth
  cdef str suffix
  cdef double M_pres, dM, dM_dz
  cdef np.ndarray[np.float64_t,ndim=1] z, M_star, M_dots, M_smooth, M_burst, M_b_dots, M_b_smooth, M_steady, M_s_dots, M_s_smooth
  cdef np.ndarray[np.float64_t,ndim=2] SF

  z = np.linspace(15., 60., 901)      #Redshift array
  all_inputs = np.load('./Deltas/All_Inputs.npy', allow_pickle=True)
  print(len(all_inputs))              #Load in all LW/vbc combos

  for i in range(0, len(all_inputs)):                 #Loop through all of the runs and smooth their PopIII
      suffix = str(round(all_inputs[i][0],2)) + '_' + str(int(all_inputs[i][1])) + special   #Suffix for this LW/vbc combo
      print(i, suffix)
      try:
#        SF = np.load('./Deltas/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy', allow_pickle=True)  #Load in SF data
        SF = np.load('/fs/scratch/PJS0312/coltonf96/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy', allow_pickle=True)
      except:
        try:
#          SF = np.load('./Deltas/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy')    #Sometimes it decides not to be pickled
          SF = np.load('/fs/scratch/PJS0312/coltonf96/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy')
        except:
          print('No file for ', suffix, i)
          continue
      if len(SF) > 5.5:
        print('Already smoothed')
        continue
      M_burst, M_steady, M_star = SF[0,:], SF[1,:], SF[2,:]                             #Isolate each row of data; bursty PopII, steady PopII, PopIII
      M_burst, M_steady = np.flipud(np.cumsum(np.flipud(M_burst))), np.flipud(np.cumsum(np.flipud(M_steady)))   #Get cumulative values for PopII
      M_b_dots, M_s_dots, M_dots = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z)) #Initialize array to record where SF(z) > SF(z+dz)
      for iz in range(0, len(z)-1):                   #Now step through time to populate dots arrays
        if M_burst[iz] > M_burst[iz+1] or iz == 0:    #If this bursty M_star(z) is > than the previous M_star(z+dz)...
          M_b_dots[iz] = M_burst[iz]                  #Record it in M_dots, otherwise it remains 0
        if M_steady[iz] > M_steady[iz+1] or iz == 0:  #Repeat for other rows (steady SF)
          M_s_dots[iz] = M_steady[iz]
        if M_star[iz] > M_star[iz+1] or iz == 0:      #And for PopIII SF
          M_dots[iz] = M_star[iz] 
      #Above is getting M_dots for both stellar populations, below is smoothiing between the two ----------------------------
      M_b_smooth, M_s_smooth, M_smooth = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z))   #Initialize smoothed SF arrays
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
        if M_b_dots[iz] > 1e-15:                  #Only when M_dots has a nonzero value...
          M_pres = M_b_dots[iz]                   #Record it and begin the smoothing loop
          for ii in range(1, len(z)-iz-1):        #Step through, at most, the rest of the z array 
            if M_b_dots[iz+ii] > 1e-15:           #Find the next z where M_dot(z) is nonzero...
              iz_smooth = ii                      #Record it & break out of the for loop
              break
            if sum(M_b_dots[iz+ii:]) < 1e-15:     #Whenever we reach the last dot...
              iz_smooth = 1                       #Use dz = 0.05 as the last iz_smooth value & break out of loop
              break
          dM = M_pres - M_b_dots[iz+iz_smooth]    #Now determine how much M_star changes from M_pres to the next nonzero value
          dM_dz = dM/iz_smooth                    #Divide by # of smoothing steps to get smoothing value for each z step
          for ii in range(0, iz_smooth):          #Now step back through z array to that next nonzero value
            M_b_smooth[iz+ii] = M_pres - ii*dM_dz #And populate M_smooth with step_size*N_step
        #Above is PopIII 'if' statement for smoothing, below is PopII 'if' statement -----------------------------------------
        if M_s_dots[iz] > 1e-15:                  #Only when M_dots has a nonzero value...
          M_pres = M_s_dots[iz]                   #Record it and begin the smoothing loop
          for ii in range(1, len(z)-iz-1):        #Step through, at most, the rest of the z array 
            if M_s_dots[iz+ii] > 1e-15:           #Find the next z where M_dot(z) is nonzero...
              iz_smooth = ii                      #Record it & break out of the for loop
              break
            if sum(M_s_dots[iz+ii:]) < 1e-15:     #Whenever we reach the last dot...
              iz_smooth = 1                       #Use dz = 0.05 as the last iz_smooth value & break out of loop
              break
          dM = M_pres - M_s_dots[iz+iz_smooth]    #Now determine how much M_star changes from M_pres to the next nonzero value
          dM_dz = dM/iz_smooth                    #Divide by # of smoothing steps to get smoothing value for each z step
          for ii in range(0, iz_smooth):          #Now step back through z array to that next nonzero value
            M_s_smooth[iz+ii] = M_pres - ii*dM_dz #And populate M_smooth with step_size*N_step
      M_smooth = np.nan_to_num(M_smooth)          #Remove NaNs from smoothed SF array 
      M_b_smooth = np.nan_to_num(M_b_smooth)      #For both stellar populations & both PopII
      M_s_smooth = np.nan_to_num(M_s_smooth)      #Overwrite SF file with smoothed values
#      np.save('./Deltas/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy', np.row_stack((M_burst, M_b_smooth, M_steady, M_s_smooth, M_star, M_smooth)))
      np.save('/fs/scratch/PJS0312/coltonf96/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy', np.row_stack((M_burst, M_b_smooth, M_steady, M_s_smooth, M_star, M_smooth)))

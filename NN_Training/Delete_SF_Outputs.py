import numpy as np
import os

def Delete(delta, start_i=0, special=''):   # MAKE SURE THAT YOU RUN COMBINE_SF_OUTPUTS.PY FIRST ------------------------
    all_inputs = np.load('./Deltas/All_Inputs.npy', allow_pickle=True)  #Load in all input combinations
    for i in range(start_i, len(all_inputs)):                           #Loop through all input combos
        inputs = all_inputs[i]                                          #Current LW/vbc combo parameters
        suffix = str(round(inputs[0],2)) + '_' + str(int(inputs[1])) + special  #Use parameters to make a filename suffix
        path = './Deltas/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy'   #Path to the SAM outputs
#        path = '/fs/scratch/PJS0312/coltonf96/Delta_' + str(int(delta)) + '/SF_' + suffix + '.npy'
        try:
          os.remove(path)   #Delete file
        except:
          print(OSError)    #Or print error if already deleted
        continue

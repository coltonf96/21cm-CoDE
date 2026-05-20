import numpy as np

def Combine(delta, special=''):
    all_inputs = np.load('./Deltas/All_Inputs_0.npy', allow_pickle=True)  #Load in all LW/v_bc combos
    print(len(all_inputs))
    all_outputs= []
    for i in range(0, len(all_inputs)):   #Step through all input combos
        inputs = all_inputs[i]            #Current combo of J_LW/v_bc inputs
        print(inputs)
        suffix = str(round(inputs[0],2)) + '_' + str(int(inputs[1]))                  #Create SAM suffix from input values
        try:
#            SF = np.load('./Deltas/Delta_' + str(int(delta)) + '/SF_' + suffix + special +'.npy')  #And load in SF history from that SAM run
            SF = np.load('/fs/scratch/PJS0312/coltonf96/Delta_' + str(int(delta)) + '/SF_' + suffix + special +'.npy')
            all_outputs.append(SF)            #Append SF output to list of outputs
        except:
            continue
    all_outputs = np.array((all_outputs)) #Convert grand list into an array & save it
    print(all_outputs[0])
    print(all_outputs.shape)
    np.save('./Deltas/Delta_' + str(int(delta)) + '/All_SF_Output_Data' + special + '.npy', all_outputs, allow_pickle=True)

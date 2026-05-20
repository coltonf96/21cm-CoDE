import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

def NN(delta, special='', N_EPOCHS=10000, N_nodes=50):
  z = np.linspace(15., 60., 901)
  t = (0.93e9)*(((1.+z)/7.)**(-1.5))
  array_path = './Deltas/Delta_' + str(int(delta))

  all_data = np.load(array_path + '/PopIII_Box_Data' + special + '.npy', allow_pickle=True)
  z_III = all_data[:,0]         #Current z steps
  log_N_III = all_data[:,1]     #Log of the number of PopIII SF events
  log_M = all_data[:,2]         #Log of integral of M_crit up to prev step
  log_J = all_data[:,3]         #Log of integral of J_LW up to prev step
  vbcs = 30. * all_data[:,4] * ((1. + z_III)/1060.)  #Streaming velocities as f(z)
  log_J_LW = all_data[:,5]      #Log of current LW values
  log_MMH = all_data[:,6]       #Log of present MMH
  output = log_N_III            #Assign log_N_III to be the Y we are emulating

  class SimpleNet(nn.Module):   #This is the Neural Net ----------------------------------------------------------
    def __init__(self):
        super(SimpleNet, self).__init__()       #An apparently necessary line -- super("ClassName", self).__init__()
        self.fc1 = nn.Linear(5, N_nodes)        #5 to 50 variables
        self.fc2 = nn.Linear(N_nodes, N_nodes)  #These 50 each talk to the next 50
        self.fc3 = nn.Linear(N_nodes, 1)        #And then gives one output (the star formation)

    def forward(self, x):         #Forward pass
        x = F.relu(self.fc1(x))   #Passes x through self.fc1 in above function : 5-50
        x = F.relu(self.fc2(x))   #Then through self.fc2 : 50-50
        x = self.fc3(x)           #Finally, back to one output : 50-1
        return x

  device = ('cuda' if torch.cuda.is_available()
    else 'cpu')                   #Try to use a GPU if possible
  bs, lr, wd = 100, 8e-5, 1e-5    #Batch size, learning rate, weight decay

  model = SimpleNet().to(device=device)       #Define model to be the class with NN & forward pass
  try:                                        #Try loading this model if it's been ran before
    model.load_state_dict(torch.load(array_path + '/Emul_' + str(N_EPOCHS) + '_PopIII' + special + '.pt0000'))
    model.eval()
  except:
    print('Did not load')                     #Otherwise, begin training model
    criterion = nn.MSELoss()                  #Mean squared error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
    loss_evo = np.zeros(N_EPOCHS)             #Initialize loss array
    for epoch in range(N_EPOCHS):             #Step through the epochs and train the NN
        ids = np.arange(len(output))
        np.random.shuffle(ids)                #Randomly shuffle indicies & make tensor of shuffled input values
        inputs0 = torch.from_numpy(np.column_stack((log_M[ids], log_J[ids], vbcs[ids], log_J_LW[ids], log_MMH[ids])))
        new_output = output[ids]              #Shuffle all outputs as well
        epoch_loss = 0.0                      #Reset Loss for this epoch
        for i in range(0, len(output), bs):   #Now step through the batches of data
            inputs = inputs0[i:i+bs]                      #The next batch of inputs
            labels = torch.tensor(new_output[i:i+bs])     #And use new_outputs when shuffled
            optimizer.zero_grad()                         #Zero the gradients
            outputs = model(inputs.float())                                 #Forward pass
            loss = criterion(outputs.float(), labels.unsqueeze(1).float())  #Compute the loss
            loss.backward()                                                 #Backward pass
            optimizer.step()                                                #Update the weights
            epoch_loss += loss.item()                                       #Add loss to running total
        print("Epoch: {} -- Loss: {}".format(epoch, epoch_loss/len(labels)))
        loss_evo[epoch] = epoch_loss/len(labels)
        if epoch % 100 == 0 and epoch > 1:
          np.save(array_path + '/Loss_' + str(N_EPOCHS) + '_PopIII' + special + '.npy', loss_evo)
    torch.save(model.state_dict(), array_path + '/Emul_' + str(N_EPOCHS) + '_PopIII' + special + '.pt')
    np.save(array_path + '/Loss_' + str(N_EPOCHS) + '_PopIII' + special + '.npy', loss_evo)   #Save the trained model & loss

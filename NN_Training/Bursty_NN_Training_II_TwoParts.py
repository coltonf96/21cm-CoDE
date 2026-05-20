import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

def NN(delta, special='', N_EPOCHS=1000, N_nodes=50):
  z = np.linspace(15., 60., 901)
  t = (0.93e9)*(((1.+z)/7.)**(-1.5))
  array_path = './Deltas/Delta_' + str(int(delta))

  all_data_b = np.load(array_path + '/PopII_Box_Data' + special + '_Bursty.npy', allow_pickle=True)
  all_data_s = np.load(array_path + '/PopII_Box_Data' + special + '_Steady.npy', allow_pickle=True)
  z_II_b = all_data_b[:,0]      #Current z steps
  log_SF_b = all_data_b[:,1]    #Log of the current PopII stellar mass due to bursty SF
  log_N_tb = all_data_b[:,2]    #Integral of M_III,200 up to a delay time ago w.r.t. t
  log_crit_b = all_data_b[:,3]  #M_Crit t_delay ago
  output_b = log_SF_b           #Assign log_SF to be the Y we are emulating

  z_II_s = all_data_s[:,0]      #Repeat for steady SF
  log_SF_s = all_data_s[:,1]
  log_N_ts = all_data_s[:,2]
  log_crit_s = all_data_s[:,3]
  log_MMH = all_data_s[:,4]     #Log of most massive halo by z (main diff)
  output_s = log_SF_s

  class SimpleNet_b(nn.Module):   #This is the Neural Net ----------------------------------------------------------
    def __init__(self):
        super(SimpleNet_b, self).__init__()     #An apparently necessary line -- super("ClassName", self).__init__()
        self.fc1 = nn.Linear(2, N_nodes)        #2 to 50 variables
        self.fc2 = nn.Linear(N_nodes, N_nodes)  #These 50 each talk to the next 50
        self.fc3 = nn.Linear(N_nodes, 1)        #And then gives one output (the star formation)
    def forward(self, x):                  #Forward pass
        x = F.relu(self.fc1(x))   #Passes x through self.fc1 in above function : 2-50
        x = F.relu(self.fc2(x))   #Then through self.fc2 : 50-50
        x = self.fc3(x)           #Finally, back to one output : 50-1
        return x

  class SimpleNet_s(nn.Module):   #This is the Neural Net ----------------------------------------------------------
    def __init__(self):
        super(SimpleNet_s, self).__init__()     #An apparently necessary line -- super("ClassName", self).__init__()
        self.fc1 = nn.Linear(3, N_nodes)        #2 to 50 variables
        self.fc2 = nn.Linear(N_nodes, N_nodes)  #These 50 each talk to the next 50
        self.fc3 = nn.Linear(N_nodes, 1)        #And then gives one output (the star formation)
    def forward(self, x):                  #Forward pass
        x = F.relu(self.fc1(x))   #Passes x through self.fc1 in above function : 2-50
        x = F.relu(self.fc2(x))   #Then through self.fc2 : 50-50
        x = self.fc3(x)           #Finally, back to one output : 50-1
        return x

  device = ('cuda' if torch.cuda.is_available()
    else 'cpu')                   #Try to use a GPU if possible
  bs, lr, wd = 100, 8e-5, 1e-5    #Batch size, learning rate, weight decay

  model_b = SimpleNet_b().to(device=device)   #Define model to be the class with NN & forward pass
  model_s = SimpleNet_s().to(device=device)   #Do so for both steady & bursty
  criterion_b, criterion_s = nn.MSELoss(), nn.MSELoss()           #Mean squared errors
  optimizer_b = torch.optim.Adam(model_b.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
  optimizer_s = torch.optim.Adam(model_s.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=wd)
  loss_evo_b, loss_evo_s = np.zeros(N_EPOCHS), np.zeros(N_EPOCHS) #Initialize loss arrays
  for epoch in range(N_EPOCHS):               #Step through the epochs and train the NN
      ids_b, ids_s = np.arange(len(output_b)), np.arange(len(output_s))
      np.random.shuffle(ids_b)                                    #Randomly shuffle indicies & make tensor of shuffled input values
      np.random.shuffle(ids_s)
      inputs0b = torch.from_numpy(np.column_stack((log_N_tb[ids_b], log_crit_b[ids_b])))
      inputs0s = torch.from_numpy(np.column_stack((log_N_ts[ids_s], log_crit_s[ids_s], log_MMH[ids_s])))
      new_output_b, new_output_s = output_b[ids_b], output_s[ids_s]   #Shuffle all outputs as well
      epoch_loss_b, epoch_loss_s = 0., 0.                             #Reset Loss for this epoch
      for i in range(0, len(output_b), bs):             #Now step through the batches of data for Bursty SF
          inputs_b = inputs0b[i:i+bs]                   #The next batch of inputs
          labels_b = torch.tensor(new_output_b[i:i+bs]) #And use new_outputs when shuffled
          optimizer_b.zero_grad()                       #Zero the gradients
          outputs_b = model_b(inputs_b.float())         #Forward pass
          loss_b = criterion_b(outputs_b.float(), labels_b.unsqueeze(1).float())  #Compute the loss
          loss_b.backward()                             #Backward passe
          optimizer_b.step()                            #Update the weights
          epoch_loss_b += loss_b.item()                 #Add loss to running total
      for i in range(0, len(output_s), bs):             #Now repeat for steady SF
          inputs_s = inputs0s[i:i+bs]                   #The next batch of inputs
          labels_s = torch.tensor(new_output_s[i:i+bs]) #And use new_outputs when shuffled
          optimizer_s.zero_grad()                       #Zero the gradients
          outputs_s = model_s(inputs_s.float())         #Forward pass
          loss_s = criterion_s(outputs_s.float(), labels_s.unsqueeze(1).float())  #Compute the loss
          loss_s.backward()                             #Backward passes
          optimizer_s.step()                            #Update the weights
          epoch_loss_s += loss_s.item()                 #Add loss to running total
      print("Bursty -- Epoch: {} -- Loss: {}".format(epoch, epoch_loss_b/len(labels_b)))
      print("Steady -- Epoch: {} -- Loss: {}".format(epoch, epoch_loss_s/len(labels_s)))
      loss_evo_b[epoch] = epoch_loss_b/len(labels_b)
      loss_evo_s[epoch] = epoch_loss_s/len(labels_s)
      if epoch % 100 == 0 and epoch > 1:
        np.save(array_path + '/Loss_' + str(N_EPOCHS) + '_PopII' + special + '_Bursty.npy', loss_evo_b)
        np.save(array_path + '/Loss_' + str(N_EPOCHS) + '_PopII' + special + '_Steady.npy', loss_evo_s)
  torch.save(model_b.state_dict(), array_path + '/Emul_' + str(N_EPOCHS) + '_PopII' + special + '_Bursty.pt')
  torch.save(model_s.state_dict(), array_path + '/Emul_' + str(N_EPOCHS) + '_PopII' + special + '_Steady.pt')
  np.save(array_path + '/Loss_' + str(N_EPOCHS) + '_PopII' + special + '_Bursty.npy', loss_evo_b)             #Save the trained models & losses
  np.save(array_path + '/Loss_' + str(N_EPOCHS) + '_PopII' + special + '_Steady.npy', loss_evo_s)

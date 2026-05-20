import numpy as np
import sys
import Box_NN_Training_II as B
import Bursty_NN_Training_II_Bursty as BB
import Bursty_NN_Training_II_Steady as BS

waste = int(sys.argv[1])
delta = int(sys.argv[2])
N_nodes = 50 
N_ep = 1000
burst = 1

if burst == 0:
    special = '_Fiducial_3b'
    B.NN(delta, special, N_ep)
else:
    special = '_Mcrit'
    BB.NN(delta, special)
    BS.NN(delta, special)

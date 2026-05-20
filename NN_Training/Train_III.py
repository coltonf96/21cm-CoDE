import numpy as np
import sys
import Box_NN_Training_III as B

waste = int(sys.argv[1])
delta = int(sys.argv[2])
N_nodes = 50 
N_ep = 10000
bursty = 1
special = '_Fiducial_3b'
if bursty == 1:
    special = '_Mcrit'
B.NN(delta, special)#, N_ep, N_nodes)

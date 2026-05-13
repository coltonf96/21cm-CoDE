import numpy as np
cimport numpy as np
cimport cython

def Get_M_Crit(double J, double v_bc, double z, double J0_M_crit, str special):
    cdef double a_0, M_20_0, beta_1, beta_2, beta_3, gamma_1, gamma_2, gamma_3, J_0, v_0, Jv_0, alpha, M_20, M_crit_1

    M_20_0, beta_1, beta_2, beta_3 = 1.96e5, 0.8, 1.83, -0.06
    gamma_1, gamma_2, gamma_3 = 0.36, -0.62, 0.13               #Various parameters from Kulkarni et al. 2021 M_crit fit
    a_0, J_0, v_0, Jv_0 = 1.64, 1.0, 30.0, 3.0

    alpha = a_0 * ((1. + J/J_0)**gamma_1) * ((1. + v_bc/v_0)**gamma_2) * ((1. + (J*v_bc)/Jv_0)**gamma_3)
    if special == '_Hazlett':
        M_20_0 *= 2.25
        alpha -= 0.8
        J0_M_crit *= 2.25 * ((1.+z)/21.)**0.8
    M_20 = M_20_0 * ((1. + J/J_0)**beta_1) * ((1. + v_bc/v_0)**beta_2) * ((1. + (J*v_bc)/Jv_0)**beta_3)
    M_crit_1 = M_20 * ((1. +z)/21.)**-alpha
    if M_crit_1 < J0_M_crit:
        M_crit_1 = J0_M_crit
    return(M_crit_1)

def Get(int J_i, int sigma_v, special=''):
    cdef int iz
    cdef double v_bc, M_a, M_H2
    cdef np.ndarray[np.float64_t,ndim=1] z, M_crit_z, J_0, J_z, vbcs

    z = np.linspace(15., 60., 901)      #Redshift values
    M_crit_z = np.zeros(len(z))         #Initialize M_crit(z) array
    vbcs = 30.*np.linspace(0., 3., 301) #Array of vbc values 0-90 km/s
    v_bc = vbcs[sigma_v]                #Streaming velocity 
    J_0 = np.load('./J_LW_0/J_0/J_0_' + str(round(v_bc/30.,2)) + '.npy')#Array of LW = 0 M_Crit values for this sigma_vbc
    J_z = np.load('./J_LW_0/Training_Js/J_' + str(J_i) + '.npy')        #Chosen J_LW(z) for this v_bc bin
    for iz in range(899, -1, -1):                                   #Loop through time
        M_a = 5.4e7*(((1.+z[iz])/11.)**-1.5)                        #Atomic cooling mass
        M_H2 = Get_M_Crit(J_z[iz], v_bc, z[iz], J_0[iz], special)   #H2 cooling mass (Kulkarni et al. 2021)
        M_crit_z[iz] = min(M_a, M_H2)                               #Record M_crit(z) as minimum of the two
    print(M_crit_z[0:10])
    np.save('./M_Crits/M_crit_' + str(sigma_v) + '_' + str(J_i) + special + '.npy', M_crit_z)   #Save M_crit(z)

    
    
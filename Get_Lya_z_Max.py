import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(15., 60., 901)   #Redshift array
ns = np.arange(2, 24)            #Relevant hydrogen energy levels
Ens = 13.6 * (1. - ns**-2.)      #Energy of each nth transition to Ly-alpha (eV)

get = 0
if get == 1:
    z_max_all = np.zeros((len(z), len(ns)-1))  #Initialize 2D array of z_max values for each z & E
    for iz in range(0, len(z)):                #Loop through redshifts
        z_pres = round(z[iz],2)                #Present redshift
        print(z_pres)
        for N in range(0, len(ns)-1):                       #Loop through each E level to populate array
            z_max = (Ens[N+1]/Ens[N]) * (1.+z_pres) - 1.    #Calculate z_max (EQ 58 of Mittal & Furlanetto 24)
            print(ns[N], round(Ens[N],2), round(z_max,2))
            z_max_all[iz][N] = z_max
    print(z_max_all)
    np.save('./Ly_alpha_z_max.npy', z_max_all)

plot = 1 
if plot == 1:
    plt.figure()
    colors = plt.cm.jet(np.linspace(0,1,len(z)))
    z_max_all = np.load('./Ly_alpha_z_max.npy')
    for iz in range(0, len(z), 50):
        z_pres = round(z[iz],2)
        labels = '_nolegend_'
        if iz % 100 == 0:
            labels = str(z_pres)
            print(z_pres)
        z_maxs = z_max_all[iz]
        plt.plot(ns[:-1], z_maxs, color=colors[iz], label=labels)
    plt.xlabel('H energy level -- n')
    plt.ylabel('Max Redshift')
    plt.legend()
    plt.savefig('Ly-a_Max_zs.pdf')

    plt.figure()
    z_max_z = np.zeros(len(z))
    for iz in range(0, len(z)):
        z_max_z[iz] = z_max_all[iz][0]
    plt.plot(z, z_max_z, label='Max z')
    plt.plot(z, z, ls='--', color='black', label='z = z')
    plt.xlabel('Current Redshift')
    plt.ylabel('Max Redshift (n=2)')
    plt.legend()
    plt.savefig('./Ly-a_Max_z.pdf')

    plt.figure()
    diff = z_max_z - z
    print(z[::50])
    print(diff[::50])
    plt.plot(z, diff)
    plt.axhline(0., ls='--', color='black')
    plt.xlabel('Current Redshift')
    plt.ylabel('dz to Max Redshift')
    plt.savefig('./Ly-a_Max_z2.pdf')

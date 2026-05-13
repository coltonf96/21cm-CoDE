import numpy as np

def get_Dz(z):  #User defined code for populating D array
  a = 1./(1.+z)
  ap = np.linspace(0, a, 10000)
  why = np.sqrt(oml*a**3.+omm) / a**(3./2.) * np.trapz(ap**(3./2.)/(oml*ap**3.+omm)**(3./2.), ap)
  return why/1.125940274656245/0.877787481277015

#User-defined parameters to run code
special = '_Dark_Ages'
get = [1,1]
CMB = 0

z = np.linspace(60., 200., 141)
print(z)
#z = np.linspace(15., 60., 901)      #Fiducial z array
if CMB == 1:
  z = np.linspace(15., 1060., 1046) #z array extending to Recombination
t = (0.93e9)*(((1.+z)/7.)**(-1.5))  #Hubble time array
omm = 0.32                          #Cosmo density parameters
oml = 1.- omm

if get[0] == 1:                 #If we need to calculate D(z) for z array
  D = np.zeros(len(z))          #Initialize D(z) array
  for iz in range(0, len(z)):   #Loop through time
    z_pres = round(z[iz], 2)    #Current redshift
    D[iz] = get_Dz(z_pres)      #Calculate D(z_pres)
    print(z_pres, D[iz])
  np.save('./Dz' + special + '.npy', D)

if get[1] == 1:                 #If we need to calculate the derivative of D(z)
  D_dot = np.zeros(len(z))      #Initialize derivative array
  for iz in range(0, len(z)-1): #Step through time
    z_pres = round(z[iz], 2)    #Current redshift
    del_D = D[iz] - D[iz+1]     #Change in D(z) over this step
    del_t = t[iz] - t[iz+1]     #Change in Hubble time (yr)
    D_dot[iz] = del_D/del_t     #Record derivative 
    print(z_pres, D_dot[iz])
  np.save('./Dz_dot' + special + '.npy', D_dot)

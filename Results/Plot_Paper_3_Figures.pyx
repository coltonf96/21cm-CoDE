import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.colors as cols
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
cimport numpy as np
cimport cython

def Smoothing(np.ndarray[np.float64_t,ndim=1] array, int smoothing=10):
    cdef int i, ix
    cdef float sum_vals
    cdef np.ndarray[np.float64_t,ndim=1] smooth_array, z

    z = np.linspace(15., 60., 901)
    array = np.nan_to_num(array)
    smooth_array = np.zeros(len(array))
    smooth_array[0:smoothing] = sum(array[0:smoothing])/float(smoothing)
    for i in range(smoothing, len(z)-smoothing):
        sum_vals = 0.0
        for ix in range(-smoothing, smoothing+1):
            sum_vals += array[i+ix]
        smooth_array[i] = sum_vals/(2.0*smoothing + 1.0)
    return(smooth_array)

def Plot(list specials, np.ndarray[np.int_t,ndim=1] plot, int overwrite=0, ref_plot=0):
    cdef int N_side, N_slice, i, j, iz, iz_i, iz_II_B, iz_II_S, iz_III_b, iz_II, iz_III
    cdef str fid_special, box_special, m100_special, m200_special, m400_special, crit_special, delay_special, int_special, path, path_0, path_100, path_200, path_400, path_crit, path_del, path_i, path_21, path_021, path_100_21, path_200_21, path_400_21, path_crit21, path_del21, path_i21
    cdef list paths, paths_21, labels, colors, styles
    cdef double V_com, z_pres, J_a_min, J_a_max, SF_min, SF_max, T_S_min, T_S_max, T_b_min, T_b_max, mean_II_B, mean_II_S, mean_III_b, mean_II, mean_III
    cdef np.ndarray[np.float64_t,ndim=1] z, t, T_cmb, T_S_avg, T_S_SD, T_b_avg, T_b_SD, T_K_avg, T_K_SD, T_S_avg_i, T_S_SD_i, T_b_avg_i, T_b_SD_i, T_K_avg_i, T_K_SD_i, T_S_avg_b, T_S_SD_b, T_b_avg_b, T_b_SD_b, T_K_avg_b, T_K_SD_b, z_Edges, T_b_Edges, z_plot_a, z_plot_b, z_plot, P, k, N, Pi, ki, Ni, avg_II, SD_II, avg_III, SD_III, avg_II_i, SD_II_i, avg_III_i, SD_III_i, avg_II_b, SD_II_b, avg_III_b, SD_III_b, avg_LW, SD_LW, avg_LW_i, SD_LW_i, avg_LW_b, SD_LW_b, avg_a, SD_a, avg_a_i, SD_a_i, avg_a_b, SD_a_b, avg_X, avg_X_i, avg_X_b
    cdef np.ndarray[np.float64_t,ndim=2] T_S_data, T_b_data, T_K_data, T_S_data_i, T_b_data_i, T_K_data_i, T_S_data_b, T_b_data_b, T_K_data_b, Edges_data, T_b_M22_data, T_b_GJ_data, data_P, data_k, data_N, data_Pi, data_ki, data_Ni, data_alpha, data_star, data_TS, data_Tb, perc_diff_II, perc_diff_III, perc_diff_II_Bb, perc_diff_II_Sb, perc_diff_III_b, pop_II, pop_III, pop_II_i, pop_III_i, pop_II_b, pop_III_b, J_LW, J_LW_i, J_LW_b, J_a, J_a_i, J_a_b, J_X, J_X_i, J_X_b
    cdef np.ndarray[np.float64_t,ndim=4] J_alpha_all, SFR_II_all_0, SFR_II_all, SFR_III_all, T_S_all, T_b_all, SFR_all

    z = np.linspace(15., 60., 901)      #Redshift array
    t = (0.93e9)*(((1.+z)/7.)**(-1.5))  #Hubble time array
    N_side, V_com = 64, 27.             #N_cells per side of sim volume & cell volume [cMpc^3] 
    N_slice = int(N_side/2)             #Middle slice of cube

    fid_special, box_special, m100_special, m200_special, m400_special = specials[0], specials[1], specials[2], specials[3], specials[4]
    crit_special, delay_special, int_special = specials[5], specials[6], specials[7]    #Denote specials for each run
    z_plot = np.array([40., 35., 30., 25., 20., 15])            #Redshifts at which we'll plot maps
    path = './Box_' + str(N_side) + fid_special + '/'           #Path to all fiducial results
    path_0 = './Box_' + str(N_side) + box_special + '/'         #Path to all Paper2 results
    path_100 = './Box_' + str(N_side) + m100_special + '/'      #Path to results with M_PopIII = 100 M_sun
    path_200 = './Box_' + str(N_side) + m200_special + '/'      #To results with M_PopIII = 200 M_sun
    path_400 = './Box_' + str(N_side) + m400_special + '/'      #And results with M_PopIII = 400 M_sun
    path_crit = './Box_' + str(N_side) + crit_special + '/'     #Results where M_crit x 2.25
    path_del = './Box_' + str(N_side) + delay_special + '/'     #Results with longer t_delay
    path_i = './Integral_' + str(N_side) + int_special + '/'    #Finally, path to all HMF integral results
    path_21, path_021, path_100_21, path_200_21, path_400_21 = path+'21cm/', path_0+'21cm/', path_100+'21cm/', path_200+'21cm/', path_400+'21cm/'
    path_crit21, path_del21, path_i21 = path_crit+'21cm/', path_del+'21cm/', path_i+'21cm/'                     #Path to all 21-cm results for each simulation 
    paths = [path, path_0, path_100, path_200, path_400, path_crit, path_del, path_i]                           #Make a list of paths to loop through when plotting
    paths_21 = [path_21, path_021, path_100_21, path_200_21, path_400_21, path_crit21, path_del21, path_i21]                #For both SF and 21-cm results
    labels = ['Fiducial', 'F25-Fid', 'Fid-100', 'Fid-200', 'Fid-400', 'Hi-$M_{\mathrm{crit}}$', 'Delayed', 'HMF Integral']  #Curve labels & colors for plotting 
    colors = ['black', 'blue', 'orange', 'green', 'violet', 'red', 'brown', 'gray']     #Be sure to change these lines when adding/deleting runs
    styles = ['solid', 'solid', '--', '--', '--', 'solid', 'solid', 'solid']            #Curve linestyles for busy plots showing all runs

    if plot[0] == 1:
        print('Plotting Globally-Averaged Temperature evolutions')
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.8,1.0]}, layout='constrained')          #Initialize plots
        for i in range(0, len(paths_21)):                                              #Loop through each run to plot their temps
            T_S_data = np.load(paths_21[i] + 'T_S_Avg_SD' + specials[i] + '.npy')      #Load in avg/SD of spin temp...
            T_b_data = np.load(paths_21[i] + 'T_b_Avg_SD' + specials[i] + '.npy')*1e3  # brightness temp in mK...
            T_K_data = np.load(paths_21[i] + 'T_k_Avg' + specials[i] + '.npy')         # & kinetic temperature
            T_S_avg, T_S_SD = T_S_data[:,0], T_S_data[:,1]
            T_b_avg, T_b_SD = T_b_data[:,0], T_b_data[:,1]  #Denote avg & SD of each
            T_K_avg, T_K_SD = T_K_data[:,0], T_K_data[:,1]
            print(labels[i] + ' min/max: ', np.min(T_b_avg), np.max(T_b_avg))       #Print T_b mins/maxs and their z's
            print('Corresponding z: ', round(z[np.argmin(np.abs(T_b_avg-np.min(T_b_avg)))],2), round(z[np.argmin(np.abs(T_b_avg-np.max(T_b_avg)))],2))
            ax[0].plot(z, T_K_avg, ls='--', c=colors[i], zorder=100-i)            #Plot T_K & its St.Dev.
            ax[0].plot(z, T_S_avg, c=colors[i], label=labels[i], zorder=100-i)    #Plot T_S & SD
            ax[1].plot(z, T_b_avg, c=colors[i], zorder=100-i)                     #And finally, plot T_b + SD
            if ref_plot == 1 and labels[i] == 'Fiducial':           #If we want to plot markers on redshifts which...
                for iz in range(0, len(z_plot)):                    #Are plotted in the following figures (maps & PS)...
                    z_pres = round(z_plot[iz],2)                    #Loop through time & plot markers on those zs
                    iz_i = np.argmin(np.abs(z-z_pres))
                    ax[0].plot(z_pres, T_S_avg[iz_i], '*', color='lime', zorder=100)
                    ax[1].plot(z_pres, T_b_avg[iz_i], '*', color='lime', zorder=100)
                    print(z_pres, T_S_avg[iz_i], T_b_avg[iz_i])
        T_cmb = 2.726 * (1.+z)                                          #CMB temperature as f(z)
        Edges_data = np.load('./EDGES_Fit.npy')                         #Load in EDGES data
        z_Edges, T_b_Edges = Edges_data[:,0], Edges_data[:,1]*1e3       #Denote the redshifts & T_b from EDGES
        z_21cm = np.flipud(np.load('../Paper3-21cmfast/Redshift_Munoz_2022.npy'))
        T_b_21cm = np.load('../Paper3-21cmfast/Results/T_b/T_b_Munoz_2022.npy') #Load in 21cmFast data
        T_b_lit = np.load('./Comparison_Tbs.npy', allow_pickle=True)            #Load in literature comparisons
        z_lit = np.load('./Comparison_Tb_Redshifts.npy', allow_pickle=True)     #Loading in their redshifts
        labels_lit = np.load('./Comparison_Tb_Labels.npy', allow_pickle=True)   #Brightness temps, and labels
        z_liu_10, z_liu_100, z_vent_w, z_vent_m, z_GJ, z_HF = z_lit             #Denote each works results
        T_liu_10, T_liu_100, T_vent_w, T_vent_m, T_GJ, T_HF = T_b_lit
        ax[0].plot(z, T_cmb, ls=':', c='black', label='$T_{\gamma}$ - CMB')        #Plot the CMB temperature 
        ax[1].plot(z_Edges, T_b_Edges, color='gray', ls='--', label='EDGES Fit')   #And plot the literature reference T_b's for comparison
        ax[1].plot(z_liu_10, T_liu_10, '-o', markersize=3, color='lightblue', label=labels_lit[0], zorder=1)
        ax[1].plot(z_liu_100, T_liu_100, '-o', markersize=3, color='lightblue', ls='--', label=labels_lit[1], zorder=1)
        ax[1].plot(z_vent_w, T_vent_w, '-o', markersize=3, color='peachpuff', label=labels_lit[2], zorder=1)
        ax[1].plot(z_vent_m, T_vent_m, '-o', markersize=3, color='peachpuff', ls='--', label=labels_lit[3], zorder=1)
        ax[1].plot(z_GJ, T_GJ, '-o', markersize=3, color='lime', label=labels_lit[4], zorder=1)
        ax[1].plot(z_HF, T_HF, '-o', markersize=3, color='lightcoral', label=labels_lit[5], zorder=1)
        ax[1].plot(z_21cm, T_b_21cm, color='dimgray', ls='-.', label='$\mathrm{Mu\~{n}oz}$+2022')
        ax[1].axhline(0., ls=':', c='black')    #Plot T_S = T_CMB line in T_b plot (T_b = 0 mK)
        ax[0].set_xlim(39.5, 15)
        ax[1].set_xlim(39.5, 13)
        ax[0].set_ylim(5., 121.)
        ax[1].set_ylim(-176., 24.)
        fig.supxlabel('Redshift',fontsize=12)
        ax[0].set_ylabel('Temperature [K]',fontsize=12)
        ax[1].set_ylabel('$\delta T_{\mathrm{b}}$ [mK]',fontsize=12)
        ax[0].legend(ncol=3, fontsize=9)
        ax[1].legend(fontsize=9)
        ax[0].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[1].tick_params(which='both', left=True, right=True, top=True, direction="in")
        fig.set_size_inches(10, 4)
        plt.tight_layout()
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Tempreature_z_Evolution.pdf')    #Save the figure
        # --------------------------------------------------------------------------------------------------------------
        plt.figure()                                                                        #Now to plot comparison T_b evolutions from the literature 
        T_b_avg_0 = np.load(paths_21[0] + 'T_b_Avg_SD' + specials[i] + '.npy')[:,0] * 1e3   #Fiducial brightness temp in mK...
        T_b_lit = np.load('./Comparison_Tbs.npy', allow_pickle=True)                        #Load in literature comparisons
        z_lit, labels_lit = np.load('./Comparison_Tb_Redshifts.npy', allow_pickle=True), np.load('./Comparison_Tb_Labels.npy', allow_pickle=True)
        z_liu_10, z_liu_100, z_vent_w, z_vent_m, z_GJ, z_HF = z_lit
        T_liu_10, T_liu_100, T_vent_w, T_vent_m, T_GJ, T_HF = T_b_lit
        plt.plot(z, T_b_avg_0, color='black', label='Fiducial')
        plt.plot(z_Edges, T_b_Edges, color='black', ls='--', label='EDGES Fit')         #Plot the literature reference T_b's for comparison
        plt.plot(z_liu_10, T_liu_10, '-o', markersize=3, label=labels_lit[0], zorder=1)
        plt.plot(z_liu_100, T_liu_100, '-o', markersize=3, label=labels_lit[1], zorder=1)
        plt.plot(z_vent_w, T_vent_w, '-o', markersize=3, label=labels_lit[2], zorder=1)
        plt.plot(z_vent_m, T_vent_m, '-o', markersize=3, label=labels_lit[3], zorder=1)
        plt.plot(z_GJ, T_GJ, '-o', markersize=3, label=labels_lit[4], zorder=1)
        plt.plot(z_HF, T_HF, '-o', markersize=3, label=labels_lit[5], zorder=1)
        plt.axhline(0., ls=':', c='black')
        plt.xlim(35, 13)
        plt.ylim(-166., 24.)
        plt.xlabel('Redshift',fontsize=12)
        plt.ylabel('$\delta T_{\mathrm{b}}$ [mK]',fontsize=12)
        plt.legend(fontsize=12)
        plt.tick_params(which='both', left=True, right=True, top=True, direction="in")
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Tempreature_z_Evolution_Comparison.pdf')
        # --------------------------------------------------------------------------------------------------------------
        fig, ax = plt.subplots(1, 3, sharex=True)                                       #Then initialize figure of ratio values
        T_S_avg_0 = np.load(paths_21[i] + 'T_S_Avg_SD' + specials[i] + '.npy')[:,0]     #Load in Fiducial T_S avg...
        T_b_avg_0 = np.load(paths_21[i] + 'T_b_Avg_SD' + specials[i] + '.npy')[:,0]*1e3 # brightness temp in mK...
        T_K_avg_0 = np.load(paths_21[i] + 'T_k_Avg' + specials[i] + '.npy')[:,0]        # & T_K for fiducial
        for i in range(1, len(paths_21)):                                               #Loop through the rest of the runs to plot their ratios
            T_S_avg = np.load(paths_21[i] + 'T_S_Avg_SD' + specials[i] + '.npy')[:,0]      #Load in avg/SD of spin temp...
            T_b_avg = np.load(paths_21[i] + 'T_b_Avg_SD' + specials[i] + '.npy')[:,0]*1e3  # brightness temp in mK...
            T_K_avg = np.load(paths_21[i] + 'T_k_Avg' + specials[i] + '.npy')[:,0]         # & kinetic temperature       
            ratio_K, ratio_S, ratio_b = T_K_avg/T_K_avg_0, T_S_avg/T_S_avg_0, T_b_avg/T_b_avg_0
            ax[0].plot(z, ratio_K, color=colors[i], label=labels[i])
            ax[1].plot(z, ratio_S, color=colors[i])    #Calculate & plot ratios
            ax[2].plot(z, ratio_b, color=colors[i])
        ax[0].axhline(1., color=colors[0], lw=0.5)
        ax[1].axhline(1., color=colors[0], lw=0.5)  #Add ratio = 1 line
        ax[2].axhline(1., color=colors[0], lw=0.5)
        ax[0].set_xlim(44, 15)
        ax[1].set_xlabel('Redshift')
        ax[0].set_ylabel('Temperature Ratios')
        ax[0].legend()
        ax[0].set_title('Kinetic')
        ax[1].set_title('Spin')
        ax[2].set_title('21-cm Brightness')
        ax[0].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[1].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[2].tick_params(which='both', left=True, right=True, top=True, direction="in")
        fig.set_size_inches(10, 4)
        plt.tight_layout()
        if overwrite == 1:      #Save ratio plot
            plt.savefig('./Paper_3_Figures/Tempreature_z_Evolution_RATIOS.pdf')

    if plot[1] == 1:
        print('Plotting Power Spectra & Ratios at Various Redshifts')
        fig, ax = plt.subplots(5, 3, sharex=True, gridspec_kw={'height_ratios': [1,0.6,0.25,1,0.6]})  #Intialize plot, 2x3 with smaller ratio panels & a blank space between the two rows
        z_plot_a, z_plot_b = np.array([40.,35.,30.]), np.array([25.,20.,15])    #Redshifts to plot (top & bottom rows)
        size, eline, caps, capt, width = len(paths), 1., len(paths), 1., 2.     #PS curve plotting parameters -- Make each subsequent run smaller than the previous to see changes
        z_21 = np.load('../Paper3-21cmfast/Redshift_Munoz_2022.npy')
        data_P21 = np.load('../Paper3-21cmfast/Results/T_b/Power_Spectra/All_Power_Munoz_2022.npy')
        data_k21 = np.load('../Paper3-21cmfast/Results/T_b/Power_Spectra/All_k_Munoz_2022.npy')     #Load in 21cmFast data
        data_N21 = np.load('../Paper3-21cmfast/Results/T_b/Power_Spectra/All_N_k_Munoz_2022.npy')
        for iz in range(0, len(z_plot_b)):
            HERA = np.load('../Paper3-21cmfast/Results/21cmSense/HERA_' + str(round(z_plot_b[iz])) + '.npy')
            k_H, S_H = HERA[:,0], HERA[:,1]
            if iz == 0:
                ax[3,iz].loglog(k_H, S_H, color='black', label='HERA')
            else:
                ax[3,iz].loglog(k_H, S_H, color='black')
        for i in range(0, len(paths)):                                          #Loop through each sim realization & plot their power spectra
            data_P = np.load(paths_21[i] + 'All_Power' + specials[i] + '.npy')  #Load in power P(k) of all z's -- Fiducial method
            data_k = np.load(paths_21[i] + 'All_k' + specials[i] + '.npy')      #Load in corresponding k bins
            data_N = np.load(paths_21[i] + 'All_N_k' + specials[i] + '.npy')    #And counts for each bin to estimate error
            if i == 0: 
                data_Pf, data_kf, data_Nf = np.copy(data_P), np.copy(data_k), np.copy(data_N)  #Denote the fiducial values for ratio plotting
            for iz in range(0, len(z_plot_a)):                      #Now loop through the first z array
                z_pres = round(z_plot_a[iz],2)                      #Current redshift
                iz_i = np.argmin(np.abs(z-z_pres))                  #Find redshift index in overall array
                P, k, N = data_P[iz_i], data_k[iz_i], data_N[iz_i]          #Denote the current power, bins, & counts
                Pf, kf, Nf = data_Pf[iz_i], data_kf[iz_i], data_Nf[iz_i]    #Also get the fiducial values for ratio plotting
                P = P*k**3./(2.*np.pi**2.)
                Pf = Pf*kf**3./(2.*np.pi**2.)
                nonz = np.nonzero(P)[0]                                     #And only plot the nonzero values of the PS
                if iz == 1:
                    label_i, label_21 = labels[i], '$\mathrm{Mu\~{n}oz}$+2022'
                else:
                    label_i, label_21 = '_nolegend_', '_nolegend_'
                ax[0,iz].errorbar(k[nonz], P[nonz], fmt='-o', yerr=P[nonz]/np.sqrt(N[nonz]), elinewidth=eline-(0.1*i), ms=2, capsize=2, capthick=capt-(0.1*i), c=colors[i], lw=width-(0.2*i), label=label_i)
                if i == len(paths)-1:
                    if z_pres < 39.0:
                        iz_21 = np.argmin(np.abs(z_21-z_pres))
                        P21, k21, N21 = data_P21[iz_21], data_k21[iz_21], data_N21[iz_21]
                        nonz21 = np.nonzero(P21)[0]
                        ax[0,iz].errorbar(k21[nonz21], P21[nonz21], fmt='^', yerr=N21[nonz21], elinewidth=0.1, ms=2, capsize=2, capthick=0.1, c='black', lw=1., mfc='gray', label=label_21, zorder=1)
                    ax[1,iz].axhline(1., c=colors[0], lw=0.8, zorder=100.)  #Add a ratio = 1 line for fiducial run
                    ax[0,iz].set_xscale("log")
                    ax[0,iz].set_yscale("log")                  #Set PS plots to be log-log scale
                    ax[0,iz].tick_params(which='both', left=True, right=True, top=True, direction="in")
                    ax[1,iz].tick_params(which='both', left=True, right=True, top=True, direction="in")
                    ax[2,iz].set_axis_off()                     #Make middle row blank
                    ax[0,iz].set_xlim(0.04, 2.)
                    ax[0,iz].set_ylim(2e-2, 1000)                 #Set power spectra y limits
                    ax[1,iz].set_ylim(0.05, 3.1)                  #And ratio y limits
                    ax[0,iz].set_title('$z$ = ' + str(z_pres))    #Label with redshift
                else:
                    ax[1,iz].loglog(k, P/Pf, '-o', ms=2, c=colors[i])     #Or plot the ratio if it's not fiducial
            for iz in range(0, len(z_plot_b)):                              #THEN loop through the second z array for the bottom row
                z_pres = round(z_plot_b[iz],2)                              #Current redshift
                iz_i = np.argmin(np.abs(z-z_pres))                          #Find redshift index in overall array
                P, k, N = data_P[iz_i], data_k[iz_i], data_N[iz_i]          #Denote the current power, bins, & counts
                Pf, kf, Nf = data_Pf[iz_i], data_kf[iz_i], data_Nf[iz_i]    #Also get the fiducial values for ratio plotting
                P = P*k**3./(2.*np.pi**2.)
                Pf = Pf*kf**3./(2.*np.pi**2.)
                nonz = np.nonzero(P)[0]                                     #And only plot the nonzero values of the PS
                ax[3,iz].errorbar(k[nonz], P[nonz], fmt='-o', yerr=P[nonz]/np.sqrt(N[nonz]), elinewidth=eline-(0.1*i), ms=2, capsize=2, capthick=capt-(0.1*i), c=colors[i], lw=width-(0.2*i))
                if i == len(paths)-1:
                    iz_21 = np.argmin(np.abs(z_21-z_pres))
                    P21, k21, N21 = data_P21[iz_21], data_k21[iz_21], data_N21[iz_21]
#                    P21 = P21*k21**3./(2.*np.pi**2.)
                    nonz21 = np.nonzero(P21)[0]
                    ax[3,iz].errorbar(k21[nonz21], P21[nonz21], fmt='^', yerr=N21[nonz21], elinewidth=0.1, ms=2, capsize=2, capthick=0.1, c='black', lw=1., mfc='gray', zorder=1)
                    ax[4,iz].axhline(1., c=colors[0], lw=0.8, zorder=100.)  #Add a ratio = 1 line for fiducial run
                    ax[3,iz].set_xscale("log")
                    ax[3,iz].set_yscale("log")                  #Set PS plots to be log-log scale
                    ax[3,iz].tick_params(which='both', left=True, right=True, top=True, direction="in")
                    ax[4,iz].tick_params(which='both', left=True, right=True, top=True, direction="in")
                    ax[3,iz].set_xlim(0.04, 2.)
                    ax[3,iz].set_ylim(2e-2, 1000)                 #Set power spectra y limits
                    ax[4,iz].set_ylim(0.05, 3.1)                  #And ratio y limits
                    ax[3,iz].set_title('$z$ = ' + str(z_pres))    #Label with redshift
                else:
                    ax[4,iz].loglog(k, P/Pf, '-o', ms=2, c=colors[i])      #Or plot the ratio if it's not the fiducial run
        fig.legend(loc='outside upper center', ncol=5)                     #Add legend outside of main plots
        ax[0,0].set_ylabel('$\Delta^{2}(k)$ [$\mathrm{mK^{2}}$]')
        ax[1,0].set_ylabel('Ratio',fontsize=10)
        ax[3,0].set_ylabel('$\Delta^{2}(k)$ [$\mathrm{mK^{2}}$]')
        ax[4,0].set_ylabel('Ratio',fontsize=10)
        ax[0,1].set_yticklabels([])
        ax[0,2].set_yticklabels([])
        ax[1,1].set_yticklabels([])
        ax[1,2].set_yticklabels([])     #Remove y axis numbers from all middle/right columns
        ax[3,1].set_yticklabels([])
        ax[3,2].set_yticklabels([])
        ax[4,1].set_yticklabels([])
        ax[4,2].set_yticklabels([])
        locmin = tck.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)   #Make sub log ticks appear
        ax[0,0].yaxis.set_minor_locator(locmin)
        ax[0,0].yaxis.set_minor_formatter(tck.NullFormatter())
        ax[0,1].yaxis.set_minor_locator(locmin)
        ax[0,1].yaxis.set_minor_formatter(tck.NullFormatter())
        ax[0,2].yaxis.set_minor_locator(locmin)
        ax[0,2].yaxis.set_minor_formatter(tck.NullFormatter())
        ax[3,0].yaxis.set_minor_locator(locmin)
        ax[3,0].yaxis.set_minor_formatter(tck.NullFormatter())
        ax[3,1].yaxis.set_minor_locator(locmin)
        ax[3,1].yaxis.set_minor_formatter(tck.NullFormatter())
        ax[3,2].yaxis.set_minor_locator(locmin)
        ax[3,2].yaxis.set_minor_formatter(tck.NullFormatter())
        fig.supxlabel('k [$\mathrm{Mpc^{-1}}$]',fontsize=12)
        fig.subplots_adjust(hspace=0., wspace=0.05)
        fig.set_size_inches(10, 7)
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Power_Spectra_and_Ratios.pdf')    #Save figure

    if plot[2] == 1:
        print('Plotting % difference vs. z')
        fig, ax = plt.subplots(1, 2)            #Initialize plot for PopII/PopIII NN %diffs by z
        xz_II, xz_III = 42., 48.                #Redshifts used in x-axes limits
        Avg_II, Avg_III = 0., 0.                #Initialize average error of all average errors for both pops
        N_II, N_III = 0, 0                      #And the number of errors over which we'll divide to get avg avg
        for i in range(0, len(paths)-1):        #Loop through main sim realizations
            perc_diff_III = 100. * np.load(paths[i] + 'Testing/RS_Perc_Diffs_III.npy')      #Load in % diffs for both stellar pops 
            if labels[i] == 'F25-Fid':
                perc_diff_II = 100. * np.load(paths[i] + 'Testing/RS_Perc_Diffs_II.npy')    #For Paper 2 fiducial with only one PopII NN
                avg_II_z, avg_III_z, SD_II_z, SD_III_z = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z)) #Initialize arrays of %diff(z)
                for iz in range(0, len(z)):
                    avg_II_z[iz], avg_III_z[iz] = np.mean(perc_diff_II[:,iz]), np.mean(perc_diff_III[:,iz]) #Loop through time and populate avg/SD %diffs
                    SD_II_z[iz], SD_III_z[iz] = np.std(perc_diff_II[:,iz]), np.std(perc_diff_III[:,iz])     #And do this for both stellar populations
                iz_II, iz_III = np.argmax(np.nonzero(avg_II_z)[0]), np.argmax(np.nonzero(avg_III_z)[0])     #Get max z with nonzero %diff for each
                mean_II, mean_III = np.mean(avg_II_z[:iz_II+1]), np.mean(avg_III_z[:iz_III+1])              #Then get the z-averaged %diff
                print(labels[i] + ' Avg % Diffs (II/III): ', mean_II, mean_III)
                Avg_II += mean_II
                Avg_III += mean_III
                N_II += 1
                N_III += 1
                ax[0].plot(z, Smoothing(avg_II_z), color=colors[i], lw=1.5, label=labels[i])
                ax[0].axhline(mean_II, xmin=(xz_II-z[iz_II])/(xz_II-15.), lw=1.5, ls=':', color=colors[i])
                if overwrite == 1:
                    np.save(paths[i] + 'Testing/Avg_Perc_Diff_II.npy', np.row_stack((avg_II_z, SD_II_z)))
                    np.save(paths[i] + 'Testing/Avg_Perc_Diff_III.npy', np.row_stack((avg_III_z, SD_III_z)))
            else:
                perc_diff_II_B = 100. * np.load(paths[i] + 'Testing/RS_Perc_Diffs_II_B.npy')             #If it's a two-phase PopII model, load in both %diffs
                perc_diff_II_S = 100. * np.load(paths[i] + 'Testing/RS_Perc_Diffs_II_S.npy')             #For bursty & steady PopII SF
                avg_II_B_z, avg_II_S_z, avg_III_z = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z)) #Initialize arrays of %error(z)
                SD_II_B_z, SD_II_S_z, SD_III_z = np.zeros(len(z)), np.zeros(len(z)), np.zeros(len(z))    #Then loop through time & populate each
                for iz in range(0, len(z)):
                    avg_II_B_z[iz], avg_II_S_z[iz], avg_III_z[iz] = np.mean(perc_diff_II_B[:,iz]), np.nanmean(perc_diff_II_S[:,iz]), np.mean(perc_diff_III[:,iz])
                    SD_II_B_z[iz], SD_II_S_z[iz], SD_III_z[iz] = np.std(perc_diff_II_B[:,iz]), np.nanstd(perc_diff_II_S[:,iz]), np.std(perc_diff_III[:,iz])
                iz_II_B, iz_II_S, iz_III = np.argmax(np.nonzero(avg_II_B_z)[0]), np.argmax(np.nonzero(avg_II_S_z)[0]), np.argmax(np.nonzero(avg_III_z)[0])
                mean_II_B, mean_II_S, mean_III = np.mean(avg_II_B_z[:iz_II_B+1]), np.mean(avg_II_S_z[:iz_II_S+1]), np.mean(avg_III_z[:iz_III+1])
                Avg_II += mean_II_B + mean_II_S
                Avg_III += mean_III
                N_II += 2
                N_III += 1
                print(labels[i] + ' Avg % Diffs (IIB/IIS/III): ', mean_II_B, mean_II_S, mean_III)
                ax[0].plot(z, Smoothing(avg_II_B_z), color=colors[i], lw=1.5, label=labels[i])
                ax[0].axhline(mean_II_B, xmin=(xz_II-z[iz_II_B])/(xz_II-15.), lw=1.5, ls=':', color=colors[i])
                ax[0].plot(z, Smoothing(avg_II_S_z), color=colors[i], lw=0.8, label=labels[i])
                ax[0].axhline(mean_II_S, xmin=(xz_II-z[iz_II_S])/(xz_II-15.), lw=0.8, ls=':', color=colors[i])
                if overwrite == 1:
                    np.save(paths[i] + 'Testing/Avg_Perc_Diff_II_Bursty.npy', np.row_stack((avg_II_B_z, SD_II_B_z)))
                    np.save(paths[i] + 'Testing/Avg_Perc_Diff_II_Steady.npy', np.row_stack((avg_II_S_z, SD_II_S_z)))
                    np.save(paths[i] + 'Testing/Avg_Perc_Diff_III.npy', np.row_stack((avg_III_z, SD_III_z)))
            ax[1].plot(z, Smoothing(avg_III_z), color=colors[i], lw=1.5, label=labels[i])
            ax[1].axhline(mean_III, xmin=(xz_III-z[iz_III])/(xz_III-15.), lw=1.5, ls=':', color=colors[i])
        Avg_II /= float(N_II)
        Avg_III /= float(N_III)
        print('Means: PopII -- ', Avg_II)
        print('PopIII --', Avg_III)
        ax[0].axhline(Avg_II, color='black', ls='--', zorder=1)
        ax[1].axhline(Avg_III, color='black', ls='--', zorder=1)
        ax[0].set_xlim(xz_II+1., 15.)
        ax[0].set_ylim(-1., 29.)
        ax[1].set_xlim(xz_III+1., 15.)
        ax[1].set_ylim(-0.65, 17.5)
        ax[0].set_title('PopII')
        ax[1].set_title('PopIII')
        custom_lines = [Line2D([0], [0], color='black', lw=1.5), Line2D([0], [0], color='black', lw=0.8), Line2D([0], [0], color='black', ls=':')]
        ax[0].legend(handles=custom_lines, labels=['Bursty', 'Steady', '$z$-Average'], fontsize=10, loc='upper left')
        ax[1].legend()
        ax[0].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[1].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[0].set_ylabel(r'$\epsilon_{\mathrm{NN}}$ [%]',fontsize=12)
        fig.supxlabel('Redshift',fontsize=12)
        fig.set_size_inches(10, 5)
        plt.subplots_adjust(wspace=0.15)
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/RS_Perc_Diff_Line.pdf')

    if plot[3] == 1:
        print('Plotting average SF & backgrounds')
        fig, ax = plt.subplots(5, 1, sharex=True, layout='constrained')
        for i in range(0, len(paths)):              #Loop through simulation realizations
            path = paths[i] + 'Figures/'            #Denote path to SF & J_LW arrays 
            pop_II, pop_III = np.load(path + 'SFRD_II_Avg_SD.npy'), np.load(path + 'SFRD_III_Avg_SD.npy')   #Load in PopII/III SFRD avgs/SDs
            J_LW, J_a = np.load(path + 'J_LW_Avg_SD.npy'), np.load(paths_21[i] + 'J_alpha_avg_SD.npy')      #Load in J_LW/alpha avgs/SDs
            avg_II, SD_II, avg_III, SD_III = pop_II[:,0], pop_II[:,1], pop_III[:,0], pop_III[:,1]           #Denote the avg/SD of the SFRDs
            avg_LW, SD_LW, avg_a, SD_a = J_LW[:,0], J_LW[:,1], J_a[:,0], J_a[:,1]                           #And the avg/SD of the J_alpha/LW
            SFRD_ratio = avg_II/avg_III
            iz_dom = np.argmin(np.abs(SFRD_ratio[:600]-1.))
            ax[0].semilogy(z, avg_III, color=colors[i], lw=0.8, label=labels[i], zorder=100-i)
            ax[1].semilogy(z, avg_II, color=colors[i], lw=0.8, zorder=100-i)
            ax[2].semilogy(z, SFRD_ratio, color=colors[i], lw=0.8, zorder=100-i)
            ax[2].axvline(z[iz_dom], color=colors[i], lw=0.5)
            print(labels[i], round(z[iz_dom],2))
            ax[3].semilogy(z, avg_LW, color=colors[i], lw=0.8, zorder=100-i)
            ax[4].semilogy(z, avg_a, color=colors[i], lw=0.8, zorder=100-i)
        z_21cm = np.array([22, 21, 20, 19, 18, 17, 16, 15, 14])
        SFRD_II_21cm = np.array([9e-6, 1.5e-5, 3e-5, 5e-5, 9e-5, 1.5e-4, 3e-4, 5e-4, 8e-4])
        SFRD_III_21cm = np.array([3.2e-5, 5e-5, 8e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 3.5e-4, 4e-4])
        J_21cm = np.load('../Paper3-21cmfast/Results/T_b/SFRDs/J_LW_Munoz_2022.npy')
        ax[0].semilogy(z_21cm, SFRD_III_21cm, ls='-.', lw=0.8, color='dimgray', label='$\mathrm{Mu\~{n}oz}$+2022')
        ax[1].semilogy(z_21cm, SFRD_II_21cm, ls='-.', lw=0.8, color='dimgray')
        ax[2].axhline(1., ls=':', lw=0.8, color='black')
        z_21cm = np.flipud(np.load('../Paper3-21cmfast/Redshift_Munoz_2022.npy'))
        ax[3].semilogy(z_21cm, J_21cm, ls='-.', lw=0.8, color='gray')
        ax[0].set_xlim(46., 15.)
        ax[0].set_ylim(2e-8, 5e-4)
        ax[1].set_ylim(9e-11, 0.009)
        locmin = tck.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        ax[2].yaxis.set_minor_locator(locmin)
        ax[2].yaxis.set_minor_formatter(tck.NullFormatter())
        ax[2].set_yticks(np.logspace(-3.,2.4,55), minor=True)
        ax[0].minorticks_on()
        ax[1].minorticks_on()
        ax[2].minorticks_on()
        ax[3].minorticks_on()
        ax[4].minorticks_on()
        ax[2].set_ylim(5e-3, 290.)
        ax[3].set_ylim(2e-4, 8)
        ax[4].set_ylim(5e-15, 2e-9)
        ax[4].set_xlabel('Redshift $z$', fontsize=11)
        ax[0].set_ylabel('$\mathrm{SFRD_{III}}$ [$\mathrm{M_\u2609}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{3}$]', fontsize=11)
        ax[1].set_ylabel('$\mathrm{SFRD_{II}}$ [$\mathrm{M_\u2609}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{3}$]', fontsize=11)
        ax[1].yaxis.set_label_position("right")
        ax[2].set_ylabel('$\mathrm{SFRD_{II}} / \mathrm{SFRD_{III}}$', fontsize=11)
        ax[3].set_ylabel('$J_{\mathrm{LW}}$ [$J_{21}$]', fontsize=11)
        ax[3].yaxis.set_label_position("right")
        ax[4].set_ylabel(r'$J_{\alpha}$ [$\mathrm{cm}^{-2}\ \mathrm{s}^{-1}\ \mathrm{Hz}^{-1}\ \mathrm{sr}^{-1}$]', fontsize=11)
        fig.legend(ncol=3, loc='outside upper center', fontsize=11)
        ax[0].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[1].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[2].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[3].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[4].tick_params(which='both', left=True, right=True, top=True, direction="in")
        fig.set_size_inches(6.5, 9.5)
        plt.subplots_adjust(hspace=0.05)
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Avg_Results_Comparison.pdf')
        print('Now getting ratios')
        fig, ax = plt.subplots(4, 1, sharex=True)
        avg_II_0, avg_III_0 = np.load(paths[0] + 'Figures/SFRD_II_Avg_SD.npy')[:,0], np.load(paths[0] + 'Figures/SFRD_III_Avg_SD.npy')[:,0]
        avg_LW_0, avg_a_0 = np.load(paths[0] + 'Figures/J_LW_Avg_SD.npy')[:,0], np.load(paths_21[0] + 'J_alpha_avg_SD.npy')[:,0]
        for i in range(1, len(paths)):              #Loop through simulation realizations
            path = paths[i] + 'Figures/'            #Denote path to SF & J_LW arrays 
            pop_II, pop_III = np.load(path + 'SFRD_II_Avg_SD.npy'), np.load(path + 'SFRD_III_Avg_SD.npy')   #Load in PopII/III SFRD avgs/SDs
            J_LW, J_a = np.load(path + 'J_LW_Avg_SD.npy'), np.load(paths_21[i] + 'J_alpha_avg_SD.npy')      #Load in J_LW/alpha avgs/SDs
            avg_II, SD_II, avg_III, SD_III = pop_II[:,0], pop_II[:,1], pop_III[:,0], pop_III[:,1]           #Denote the avg/SD of the SFRDs
            avg_LW, SD_LW, avg_a, SD_a = J_LW[:,0], J_LW[:,1], J_a[:,0], J_a[:,1]                           #And the avg/SD of the J_alpha/LW
            ax[0].semilogy(z, avg_III/avg_III_0, color=colors[i], ls=styles[i], lw=0.8, label=labels[i], zorder=100-i)
            ax[1].semilogy(z, avg_II/avg_II_0, color=colors[i], ls=styles[i], lw=0.8, zorder=100-i)
            ax[2].semilogy(z, avg_LW/avg_LW_0, color=colors[i], ls=styles[i], lw=0.8, zorder=100-i)
            ax[3].semilogy(z, avg_a/avg_a_0, color=colors[i], ls=styles[i], lw=0.8, zorder=100-i)
        ax[0].set_xlim(46., 15.)
        ax[0].axhline(1., color='black', lw=0.8)
        ax[1].axhline(1., color='black', lw=0.8)
        ax[2].axhline(1., color='black', lw=0.8)
        ax[3].axhline(1., color='black', lw=0.8)
        ax[0].set_ylim(2e-3, 21.)
        ax[1].set_ylim(1e-2, 4.2)
        ax[2].set_ylim(1e-2, 3.8)
        ax[3].set_ylim(5e-3, 3.9)
        ax[3].set_xlabel('Redshift $z$')
        ax[0].set_ylabel('$\mathrm{SFRD_{III}}$ [$\mathrm{M_\u2609}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{3}$]')
        ax[1].set_ylabel('$\mathrm{SFRD_{II}}$ [$\mathrm{M_\u2609}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{3}$]')
        ax[1].yaxis.set_label_position("right")
        ax[2].set_ylabel('$J_{\mathrm{LW}}$ [$J_{21}$]')
        ax[3].set_ylabel(r'$J_{\alpha}$ [$\mathrm{cm}^{-2}\ \mathrm{s}^{-1}\ \mathrm{Hz}^{-1}\ \mathrm{sr}^{-1}$]')
        ax[3].yaxis.set_label_position("right")
        fig.legend(ncol=4, loc='upper center')
        ax[0].minorticks_on()
        ax[1].minorticks_on()
        ax[2].minorticks_on()
        ax[3].minorticks_on()
        ax[0].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[1].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[2].tick_params(which='both', left=True, right=True, top=True, direction="in")
        ax[3].tick_params(which='both', left=True, right=True, top=True, direction="in")
        fig.set_size_inches(5, 8)
        plt.subplots_adjust(hspace=0.025)
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Avg_Results_Ratios.pdf')

    if plot[4] == 1:
        print('Plotting Lightcones')
        def nu_z(array):
            return np.flipud(1420.41 / (1.+array))      #Define functions for converting z --> observed 21-cm frequency
        def z_nu(array):
            return np.flipud((1420.41/array)-1.)        #And for converting the other way
        redshift = np.load(path_21 + 'Lightcone_z.npy')
        fig, ax = plt.subplots(6, 1, figsize=(7, 10), sharex=True, layout='constrained')      #Initialize figure of lightcones, avg T_b(z), and dimensionless PS
        rcParams.update({"font.size": 10})                 #Font size for title/axes labels
        # -- Panel 1 -- Average T_b(z) with standard deviation --
        T_b_data = np.load(path_21 + 'T_b_Avg_SD' + fid_special + '.npy') * 1e3 #Load in avg/SD T_b(z)
        T_b_avg, T_b_SD = T_b_data[:,0], T_b_data[:,1]                      #Denote the avg/SD of fiducial T_b(z)
        ax[0].plot(z, T_b_avg)
        ax[0].fill_between(z, T_b_avg-T_b_SD, T_b_avg+T_b_SD, alpha=0.3)    #Plot avg/SD of T_b(z)
        ax[0].axhline(0., ls=':', color='black')                            #And the T_CMB (T_b = 0)
        # -- Panel 2 -- Power by k-mode as f(z) --
        data_P = np.load(path_21 + 'All_Power_Fiducial.npy')    #Power for fiducial model at all z
        k_data = np.load(path_21 + 'All_k_Fiducial.npy')[0,:]   #K-modes for power spectra
        data_N = np.load(path_21 + 'All_N_k_Fiducial.npy')      #Number of counts in each k-bin
        colors2 = plt.cm.jet(np.linspace(0,1,len(k_data)))      #Colors for each k-mode
        lines = [np.column_stack([z, data_P[:,ik]*k_data[ik]**3/(2.*np.pi**2.)]) for ik in range(0, len(k_data))]   #Calculate all dimensionless PS
        line_collection = LineCollection(lines, color=colors2)  #Collect lines into one collection and plot
        ax[1].add_collection(line_collection)
        fig.colorbar(plt.cm.ScalarMappable(norm=cols.Normalize(np.min(k_data), np.max(k_data)), cmap="jet"), ax=ax[1], label='$k$-mode [$\mathrm{Mpc^{-1}}$]')
        # -- Panel 3 -- Stellar mass lightcone --
        lightcone = np.nan_to_num(np.log10(np.load(path_21 + 'Lightcone_Mstar.npy')), neginf=-1.)
        yaxis = np.linspace(0, N_side*3., lightcone.shape[0])   #Define LC y-axis from 0-BoxLength in Mpc
        cmap_kwargs = {}
        cmap_kwargs["vmin"] = 0.1   #Define the min/max values for the colorbar
        cmap_kwargs["vmax"] = 9.0
        im = ax[2].pcolormesh(redshift, yaxis, lightcone[0], cmap='viridis', shading='auto', **cmap_kwargs)
        fig.colorbar(im, ax=ax[2], label='$\log_{10}(M_{\\bigstar}/M_\u2609)$')
        # -- Panel 4 -- J_alpha lightcone --
        lightcone = np.nan_to_num(np.log10(np.load(path_21 + 'Lightcone_Ja.npy')), neginf=-1.)
        cmap_kwargs = {}
        cmap_kwargs["vmin"] = -13.3   #Define the min/max values for the colorbar
        cmap_kwargs["vmax"] = -8.7
        im = ax[3].pcolormesh(redshift, yaxis, lightcone[0], cmap='gist_rainbow', shading='auto', **cmap_kwargs)
        fig.colorbar(im, ax=ax[3], label='$\log_{10}(\ J_{\\alpha}/[J_{\\alpha}]\ )$')
        # -- Panel 5 -- T_b(z) lightcone --
        lightcone = np.load(path_21 + 'Lightcone_Tb.npy')
        cmap_kwargs = {}
        cmap_kwargs["vmin"] = -210.   #Define the min/max values for the colorbar
        cmap_kwargs["vmax"] = -5.
        im = ax[4].pcolormesh(redshift, yaxis, lightcone[0], cmap='viridis', shading='auto', **cmap_kwargs)
        fig.colorbar(im, ax=ax[4], label='$\delta T_{\mathrm{b}}(\\vec{x})$ [mK]')
        # -- Panel 6 -- Differential T_b(z) lightcone --
        lightcone = np.load(path_21 + 'Lightcone_Tb_Diff.npy')
        cmap_kwargs = {}
        cmap_kwargs["vmin"] = -120.   #Define the min/max values for the colorbar
        cmap_kwargs["vmax"] = 120.
        im = ax[5].pcolormesh(redshift, yaxis, lightcone[0], cmap='bwr', shading='auto', **cmap_kwargs)
        fig.colorbar(im, ax=ax[5], label='$\delta T_{\mathrm{b}}(\\vec{x}) - \overline{\delta T_{\mathrm{b}}}$ [mK]')
        ax0 = ax[0].secondary_xaxis('top', functions=(nu_z, z_nu))
        ax0.set_xlabel('Observed Frequency [MHz]')
        ax0.tick_params(which='both', direction="in")
        ax1 = ax[1].secondary_xaxis('top', functions=(nu_z, z_nu), zorder=1)
        ax1.tick_params(which='both', direction="in")
        ax2 = ax[2].secondary_xaxis('top', functions=(nu_z, z_nu), zorder=1)
        ax2.tick_params(which='both', direction="in")
        ax3 = ax[3].secondary_xaxis('top', functions=(nu_z, z_nu), zorder=1)
        ax3.tick_params(which='both', direction="in")
        ax4 = ax[4].secondary_xaxis('top', functions=(nu_z, z_nu), zorder=1)
        ax4.tick_params(which='both', direction="in")
        ax5 = ax[5].secondary_xaxis('top', functions=(nu_z, z_nu), zorder=1)
        ax5.tick_params(which='both', direction="in")
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels([])
        ax[0].set_xlim(49., 15.)
        ax[1].set_yscale('log')
        ax[1].set_ylim(0.09, 800.)
        ax[0].set_ylabel('$\delta T_{\mathrm{b}}$ [mK]')
        ax[1].set_ylabel('$\Delta^{2}(k)$')
        ax[2].set_ylabel('Distance [Mpc]')
        ax[3].set_ylabel('Distance [Mpc]')
        ax[4].set_ylabel('Distance [Mpc]')
        ax[5].set_ylabel('Distance [Mpc]')
        fig.supxlabel('Redshift $z$')
        ax[0].tick_params(which='both', left=True, right=True, direction="in")
        ax[1].tick_params(which='both', left=True, right=True, direction="in")
        ax[2].tick_params(which='both', left=True, right=True, direction="in")
        ax[3].tick_params(which='both', left=True, right=True, direction="in")
        ax[4].tick_params(which='both', left=True, right=True, direction="in")
        ax[5].tick_params(which='both', left=True, right=True, direction="in")
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Fiducial_Lightcones.pdf')

    if plot[5] == 1:
        print('Plotting Power at chosen k-mode & Ratios at Various z')
        fig, ax = plt.subplots(2, 4, sharex=True, gridspec_kw={'height_ratios': [1,0.6]}, layout='constrained')  #Intialize plot with smaller ratio panels
        k_plot = np.array([0.055, 0.145, 0.38, 1.])                             #K-modes for which we will plot z-evolutions
        z_21 = np.load('../Paper3-21cmfast/Redshift_Munoz_2022.npy')
        data_P21 = np.load('../Paper3-21cmfast/Results/T_b/Power_Spectra/All_Power_Munoz_2022.npy')
        data_k21 = np.load('../Paper3-21cmfast/Results/T_b/Power_Spectra/All_k_Munoz_2022.npy')     #Load in 21cmFast UVLF run for comparison
        data_N21 = np.load('../Paper3-21cmfast/Results/T_b/Power_Spectra/All_N_k_Munoz_2022.npy')
        for ik in range(0, len(k_plot)):                                            #Start by plotting HERA sensitivity curves
            sense = np.load('../Paper3-21cmfast/Results/21cmSense/HERA_' + str(round(k_plot[ik],3)) + '.npy')
            ik_21 = np.argmin(np.abs(data_k21[-1] - k_plot[ik]))                     #Also find index for 21cmFast run
            P21, k21, N21 = data_P21[:,ik_21], data_k21[:,ik_21], data_N21[:,ik_21] #Get data for this k-mode from 21cmFast
#            P21 = P21*k21**3./(2.*np.pi**2.)                                        #Get into dimensionless form
            if ik == 0:
                label_i, label_H = '$\mathrm{Mu\~{n}oz}$+2022', 'HERA'
            else:
                label_i, label_H = '_nolegend_', '_nolegend_'
            ax[0,ik].semilogy(np.linspace(15., 28., 19), sense, color='black', ls='--', label=label_H)
            ax[0,ik].semilogy(z_21, P21, lw=0.8, color='dimgray', ls='-.', label=label_i)
        for i in range(0, len(paths)):                                          #Loop through each sim realization & plot their power spectra
            data_P = np.load(paths_21[i] + 'All_Power' + specials[i] + '.npy')  #Load in power P(k) of all z's -- Fiducial method
            data_k = np.load(paths_21[i] + 'All_k' + specials[i] + '.npy')      #Load in corresponding k bins
            data_N = np.load(paths_21[i] + 'All_N_k' + specials[i] + '.npy')    #And counts for each bin to estimate error
            if i == 0: 
                data_Pf, data_kf, data_Nf = np.copy(data_P), np.copy(data_k), np.copy(data_N)  #Denote the fiducial values for ratio plotting
            for ik in range(0, len(k_plot)):                                    #Loop through k-modes
                ik_0 = np.argmin(np.abs(data_k[0] - k_plot[ik]))                #Find index of current k-mode
                P, k, N = data_P[:,ik_0], data_k[:,ik_0], data_N[:,ik_0]        #Denote z-evolution of that k-modes power & error
                Pf, kf, Nf = data_Pf[:,ik_0], data_kf[:,ik_0], data_Nf[:,ik_0]  #And get the correpsonding fiducial values for ratio
                P = P*k**3./(2.*np.pi**2.)
                Pf = Pf*kf**3./(2.*np.pi**2.)   #Get into dimensionless PS form
                if ik == 0:
                    label_i = labels[i]
                else:
                    label_i = '_nolegend_'
                ax[0,ik].semilogy(z[P>1e-8], P[P>1e-8], lw=0.8, c=colors[i], label=label_i, zorder=100-i)
                if i == 0:
                    ax[1,ik].axhline(1., c=colors[0], lw=0.8, zorder=100.)  #Add a ratio = 1 line for fiducial run
                    ax[0,ik].tick_params(which='both', left=True, right=True, top=True, direction="in")
                    ax[1,ik].tick_params(which='both', left=True, right=True, top=True, direction="in")
                    ax[1,ik].set_xlim(43., 15.)
                    ax[0,ik].set_title('$k$ = ' + str(k_plot[ik]) + ' $\mathrm{Mpc^{-1}}$')#Label with k-mode
                else:
                    ax[1,ik].semilogy(z[P>1e-8], P[P>1e-8]/Pf[P>1e-8], lw=0.8, c=colors[i]) #Or plot the ratio if it's not fiducial
        fig.legend(loc='outside upper center', ncol=5)                             #Add legend outside of main plots
        ax[0,0].set_ylabel('$\Delta^{2}(k,z)$ $[\mathrm{mK^{2}}]$',fontsize=10)    #And label axes
        ax[1,0].set_ylabel('Ratio',fontsize=10)
        ax[0,1].set_yticklabels([])
        ax[0,2].set_yticklabels([])
        ax[0,3].set_yticklabels([])
        ax[1,1].set_yticklabels([])
        ax[1,2].set_yticklabels([])
        ax[1,3].set_yticklabels([])
        ax[0,0].set_ylim(0.05, 2000.)
        ax[0,1].set_ylim(0.05, 2000.)
        ax[0,2].set_ylim(0.05, 2000.)
        ax[0,3].set_ylim(0.05, 2000.)
        ax[1,0].set_ylim(0.05, 4.)
        ax[1,1].set_ylim(0.05, 4.)
        ax[1,2].set_ylim(0.05, 4.)
        ax[1,3].set_ylim(0.05, 4.)
        fig.supxlabel('Redshift',fontsize=10)
        fig.set_size_inches(10., 4.5)
        fig.subplots_adjust(hspace=0.)
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Power_k_and_Ratios.pdf')    #Save figure

    if plot[6] == 1:
        print('Plotting Fiducial Temperature Maps at Various z')
        fig, ax = plt.subplots(4, len(z_plot)+1, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1, 0.1]}, layout='constrained')     #Make figure with 4 panels x N_redshifts
        J_alpha_all = np.load(paths_21[0] + 'J_alpha_all.npy')                                              #Load in Lyman-a background of all cells
        M_star_II_all = np.load(paths[0] + 'Mstar_II_B_all.npy') + np.load(paths[0] + 'Mstar_II_S_all.npy') #Cumulative PopII stellar mass of all cells
        M_star_III_all = np.load(paths[0] + 'Mstar_III_all.npy') * 200.             #And load in/convert PopIII M_star,200 --> [M_sun]
        T_b_all = np.load(paths_21[0] + 'T_b_all' + fid_special + '.npy')*1e3       #21-cm Brightness temperature of all cells [mK]
        T_b_data = np.load(paths_21[0] + 'T_b_Avg_SD' + fid_special + '.npy')*1e3   #And the average/SD
        T_b_avg, T_b_SD = T_b_data[:,0], T_b_data[:,1]                              #Separate the two arrays
        for iz in range(0, len(z_plot)):                    #Begin looping through redshifts of interest
            z_pres = round(z_plot[iz],2)                    #Current z
            iz_i = np.argmin(np.abs(z-z_pres))              #Find index in overall z array
            data_alpha0 = J_alpha_all[:,:,N_slice,iz_i]     #Isolate current cube for J_alpha
            data_Tb = T_b_all[:,:,N_slice,iz_i]             #And the 21-cm brightness temps
            data_Tb0 = data_Tb - T_b_avg[iz_i]              #Get the differential from the mean temperature
            print(z_pres, np.min(data_Tb0), np.mean(data_Tb0), np.max(data_Tb0))
            data_star0 = M_star_II_all[:,:,N_slice,iz_i] + M_star_III_all[:,:,N_slice,iz_i] #The total cumulative stellar mass
            data_alpha = np.nan_to_num(np.log10(data_alpha0), neginf=-100.)                 #Take the logs of the stellar masses & J_alpha
            data_star = np.nan_to_num(np.log10(data_star0), neginf=-1.)
            im0 = ax[0,iz].imshow(data_star, origin='lower', norm=cols.Normalize(vmin=0.1, vmax=8.2))
            im1 = ax[1,iz].imshow(data_alpha, origin='lower', norm=cols.Normalize(vmin=-13.3, vmax=-8.7), cmap='gist_rainbow')
            im2 = ax[2,iz].imshow(data_Tb, origin='lower', norm=cols.Normalize(vmin=-210., vmax=-5.))
            im3 = ax[3,iz].imshow(data_Tb0, origin='lower', norm=cols.Normalize(vmin=-120., vmax=120.), cmap='bwr')
            if z_pres == 15.0:
                fig.colorbar(im0, cax=ax[0,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()
                fig.colorbar(im1, cax=ax[1,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()
                fig.colorbar(im2, cax=ax[2,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()   #Plot color bars at bottom
                fig.colorbar(im3, cax=ax[3,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()
            ax[0,iz].set_title('z = ' + str(z_pres), fontsize=10)          #Label the column with the current redshift
            ax[0,iz].set_xticks([])
            ax[0,iz].set_yticks([])
            ax[1,iz].set_xticks([])
            ax[1,iz].set_yticks([])
            ax[2,iz].set_xticks([])    #Remove tick marks for this column
            ax[2,iz].set_yticks([])
            ax[3,iz].set_xticks([])
            ax[3,iz].set_yticks([])
        ax[0,0].set_ylabel('$\log_{10}(M_{\\bigstar}/M_\u2609)$', fontsize=8)
        ax[1,0].set_ylabel('$\log_{10}(\ J_{\\alpha}/[J_{\\alpha}]\ )$', fontsize=8)
        ax[2,0].set_ylabel('$\delta T_{\mathrm{b}}(\\vec{x})$ [mK]', fontsize=8)
        ax[3,0].set_ylabel('$\delta T_{\mathrm{b}}(\\vec{x}) - \overline{\delta T_{\mathrm{b}}}$ [mK]', fontsize=8)
        ax[0,6].tick_params(axis='both', which='major', labelsize=8)
        ax[1,6].tick_params(axis='both', which='major', labelsize=8)
        ax[2,6].tick_params(axis='both', which='major', labelsize=8)
        ax[3,6].tick_params(axis='both', which='major', labelsize=8)
        fig.set_size_inches(7.5, 5)
        plt.subplots_adjust(hspace=0.07, wspace=0.05)
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Temperature_Maps.pdf')

    if plot[7] == 1:
        print('Plotting Fiducial Temperature Maps at Various z')
        fig, ax = plt.subplots(5, len(z_plot)+1, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1, 0.1]}, layout='constrained')     #Make fig w 4 panels x N_redshifts
        J_alpha_all = np.load(paths_21[0] + 'J_alpha_all.npy')                                              #Load in Lyman-a background of all cells
#        M_star_II_all = np.load(paths[0] + 'Mstar_II_B_all.npy') + np.load(paths[0] + 'Mstar_II_S_all.npy') #Cumulative PopII stellar mass of all cells
#        M_star_III_all = np.load(paths[0] + 'Mstar_III_all.npy') * 200.             #And load in/convert PopIII M_star,200 --> [M_sun]
        M_star_II_all = np.load(paths[0] + 'SFR_II_all.npy')
        M_star_III_all = np.load(paths[0] + 'SFR_III_all.npy')
        T_b_all = np.load(paths_21[0] + 'T_b_all' + fid_special + '.npy')*1e3       #21-cm Brightness temperature of all cells [mK]
        T_b_data = np.load(paths_21[0] + 'T_b_Avg_SD' + fid_special + '.npy')*1e3   #And the average/SD
        T_b_avg, T_b_SD = T_b_data[:,0], T_b_data[:,1]                              #Separate the two arrays
        for iz in range(0, len(z_plot)):                    #Begin looping through redshifts of interest
            z_pres = round(z_plot[iz],2)                    #Current z
            iz_i = np.argmin(np.abs(z-z_pres))              #Find index in overall z array
            data_alpha0 = J_alpha_all[:,:,N_slice,iz_i]     #Isolate current cube for J_alpha
            data_Tb = T_b_all[:,:,N_slice,iz_i]             #And the 21-cm brightness temps
            data_Tb0 = data_Tb - T_b_avg[iz_i]              #Get the differential from the mean temperature
            data_star_II0, data_star_III0 = M_star_II_all[:,:,N_slice,iz_i], M_star_III_all[:,:,N_slice,iz_i]
            data_alpha = np.nan_to_num(np.log10(data_alpha0), neginf=-100.)                 #Take the logs of the stellar masses & J_alpha
            data_star_II, data_star_III = np.nan_to_num(np.log10(data_star_II0)), np.nan_to_num(np.log10(data_star_III0))
            try:
                print(z_pres, np.min(data_star_II[data_star_II>1e-100]), np.max(data_star_II), np.min(data_star_III[data_star_III>1e-100]), np.max(data_star_III))
            except:
                pass
            im0 = ax[0,iz].imshow(data_star_III, origin='lower', norm=cols.Normalize(vmin=-6.,vmax=-1.)) #(vmin=1.5, vmax=6.5))
            im1 = ax[1,iz].imshow(data_star_II, origin='lower', norm=cols.Normalize(vmin=-5.,vmax=0.5)) #(vmin=1.5, vmax=8.2))
            im2 = ax[2,iz].imshow(data_alpha, origin='lower', norm=cols.Normalize(vmin=-13.3, vmax=-8.7), cmap='gist_rainbow')
            im3 = ax[3,iz].imshow(data_Tb, origin='lower', norm=cols.Normalize(vmin=-210., vmax=-5.))
            im4 = ax[4,iz].imshow(data_Tb0, origin='lower', norm=cols.Normalize(vmin=-120., vmax=120.), cmap='bwr')
            if z_pres == 15.0:
                fig.colorbar(im0, cax=ax[0,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()
                fig.colorbar(im1, cax=ax[1,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()
                fig.colorbar(im2, cax=ax[2,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()   #Plot color bars at bottom
                fig.colorbar(im3, cax=ax[3,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()
                fig.colorbar(im4, cax=ax[4,iz+1], use_gridspec=True, orientation="vertical", shrink=0.8).minorticks_on()
            ax[0,iz].set_title('z = ' + str(z_pres), fontsize=10)          #Label the column with the current redshift
            ax[0,iz].set_xticks([])
            ax[0,iz].set_yticks([])
            ax[1,iz].set_xticks([])
            ax[1,iz].set_yticks([])
            ax[2,iz].set_xticks([])    #Remove tick marks for this column
            ax[2,iz].set_yticks([])
            ax[3,iz].set_xticks([])
            ax[3,iz].set_yticks([])
            ax[4,iz].set_xticks([])
            ax[4,iz].set_yticks([])
#        ax[0,0].set_ylabel('$\log_{10}(M_{\mathrm{\\bigstar,III}}/M_\u2609)$', fontsize=8)
#        ax[1,0].set_ylabel('$\log_{10}(M_{\mathrm{\\bigstar,II}}/M_\u2609)$', fontsize=8)
        ax[0,0].set_ylabel('$\log_{10}(SFR_{\mathrm{II}}/[M_\u2609 yr^{-1}])$', fontsize=8)
        ax[1,0].set_ylabel('$\log_{10}(SFR_{\mathrm{III}}/[M_\u2609 yr^{-1}])$', fontsize=8)
        ax[2,0].set_ylabel('$\log_{10}(\ J_{\\alpha}/[J_{\\alpha}]\ )$', fontsize=8)
        ax[3,0].set_ylabel('$\delta T_{\mathrm{b}}(\\vec{x})$ [mK]', fontsize=8)
        ax[4,0].set_ylabel('$\delta T_{\mathrm{b}}(\\vec{x}) - \overline{\delta T_{\mathrm{b}}}$ [mK]', fontsize=8)
        ax[0,6].tick_params(axis='both', which='major', labelsize=8)
        ax[1,6].tick_params(axis='both', which='major', labelsize=8)
        ax[2,6].tick_params(axis='both', which='major', labelsize=8)
        ax[3,6].tick_params(axis='both', which='major', labelsize=8)
        ax[4,6].tick_params(axis='both', which='major', labelsize=8)
        fig.set_size_inches(7.5, 6)
        plt.subplots_adjust(hspace=0.07, wspace=0.05)
        if overwrite == 1:
            plt.savefig('./Paper_3_Figures/Temperature_Maps_3.pdf')
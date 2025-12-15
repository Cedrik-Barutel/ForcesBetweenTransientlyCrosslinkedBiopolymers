#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:18:43 2025

@author: cedrik
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import datetime
from matplotlib.animation import FuncAnimation
import scipy
from scipy.ndimage import gaussian_filter1d


# from scipy.differentiate import derivative
from scipy import integrate
import pandas as pd
import scipy as scipy
import logging
import os
import sys

# local library
dir_local = os.path.dirname(__file__)
sys.path.append(dir_local)
lib_path = "/home/cedrik/Documents/filaments-crosslinkers-projects/lib"
sys.path.append(lib_path)

logger = logging.getLogger(__name__)
import lib_simulation as LS

color_m = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# fontsize_m = 6
# fontsize_legend_m = fontsize_m/1.5
cm = 1/2.54  # centimeters in inches

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})




#%%
dir_local = os.path.dirname(__file__)

name =  "Kucera2021_1212_4_fine_s1_k1000_N2000"
dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/EntropicForce/'+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = "fit"

bool_anim = 1
# print(name_save)




#%% loading data
tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files

x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])





#%%
n_img = 10
N_end = len(t_tasks)
extension_animation = ".mp4"
frame_per_second = 20


# %% ANIMATION
if bool_anim:
    fig = plt.figure(figsize=(9, 3), dpi=500)
    def animate(i):
        # if i%(N_end/100) == 0:
        print(i)
        plt.clf()    
        N_tot = integrate.simpson(tasks['Pab'][i]+tasks['Pa'][i]+tasks['Pb'][i])
        # plt.title(N_tot)
        plt.plot(x_tasks,tasks['Pab'][i],color = 'purple',linestyle="-",label = r"$n^{(ab)}$")
        plt.plot(x_tasks,tasks['Pa'][i],color = 'blue',linestyle="-",label = r"$n^{(a)}$")
        plt.plot(x_tasks,tasks['Pb'][i],color = 'red',linestyle="-",label = r"$n^{(b)}$")


        plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$n^{(A)}$")
        plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$n^{(B)}$")
        # plt.ylim(-0.05,1.5)
        
        # plt.hlines(0.475, -0.5 ,0.5)
        plt.gca().set_yticks((0,1))
        plt.gca().set_yticklabels((0,"$n_{max}$"))

        plt.legend(loc='upper left')
    
        
    t=np.arange(0,N_end,n_img) # New time array with only n images  
    ani = FuncAnimation(fig, animate, frames=t,
                        interval=1, repeat=False)
    #name = "D"+str(D)+"a"+str(alpha)+".gif"
    ani.save(dir_output_file+"/"+name_output_file+"_density"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)


# %% ANIMATION
if bool_anim:
    fig = plt.figure(figsize=(5, 2), dpi=200)
    def animate(i):
        if i%(N_end/100) == 0:
            print(i)
        plt.clf()    
        N_tot = integrate.simpson(tasks['Pab'][i]+tasks['Pa'][i]+tasks['Pb'][i])
        plt.title(N_tot)
        plt.plot(x_tasks,tasks['Pab'][i],color = 'purple',linestyle="-",label = r"$P^{ab}$")
        plt.plot(x_tasks,tasks['grad_mu_fA'][i]/np.max(abs( tasks['grad_mu_fA'][i] )),color = 'blue',linestyle="-",label = r"$P^{a}$")
        plt.plot(x_tasks,tasks['Pb'][i],color = 'red',linestyle="-",label = r"$P^{b}$")


        plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$\phi^A$")
        plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$\phi^B$")

        plt.hlines(0.475, -0.5 ,0.5)


        plt.legend(loc='upper left')
    
        
    t=np.arange(0,N_end,n_img) # New time array with only n images  
    ani = FuncAnimation(fig, animate, frames=t,
                        interval=1, repeat=False)
    #name = "D"+str(D)+"a"+str(alpha)+".gif"
    ani.save(dir_output_file+"/"+name_output_file+"_force"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)


#%%
i=1

plt.plot(x_tasks,tasks['Pab'][i],color = 'purple',linestyle="-",label = r"$P^{ab}$")
plt.plot(x_tasks,tasks['Pa'][i],color = 'blue',linestyle="-",label = r"$P^{a}$")
plt.plot(x_tasks,tasks['Pb'][i],color = 'red',linestyle="-",label = r"$P^{b}$")


plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$\phi^A$")
# plt.plot(x_tasks,np.gradient(tasks['f_A'][i],x_tasks), color='blue',alpha = 0.5, label = r"$\phi^A$")

plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
plt.vlines(x_tasks[255], 0, 1)
plt.vlines(x_tasks[86], 0, 1)
plt.legend()
plt.show()
print(x_tasks[86])

#%%
tasks['Pab'][0][255]
tasks['Pa'][0][130]

#%% Force calculation



# Force_ent_A = np.zeros(len(t_tasks))
# Force_visc_A = np.zeros(len(t_tasks))
# Force_el_A = np.zeros(len(t_tasks))
# Force_ent_B = np.zeros(len(t_tasks))

# Force_calc_A = np.zeros(len(t_tasks))


# concentration = np.zeros(len(t_tasks))

# overlap = np.zeros(len(t_tasks))

# for i in range(len(t_tasks)):
#     Force_ent_A[i] = tasks['F_fA_ent'][i][10]
#     Force_visc_A[i] = tasks['F_fA_vis'][i][10]
#     Force_el_A[i] = tasks['F_fA_ela'][i][10]

#     Force_ent_B[i] = tasks['F_fB_ent'][i][10]
#     concentration[i] = tasks['n_D'][i][int(len(x_tasks)/2)]
#     overlap[i] = integrate.simpson(tasks['f_D'][i],x_tasks)
    
#     f_A = np.array(tasks['f_A'][i])
#     f_D = np.array(tasks['f_D'][i])

#     dxf_D = np.gradient(f_D)
#     n_D = np.array(tasks['n_D'][i])
#     # n_D = np.minimum(f_D*0.99-1e-2,n_D)  
#     n_D = np.minimum(f_D,n_D)


#     h = 1e-4
#     dxn_D = np.gradient(n_D)

#     f_B = np.array(tasks['f_B'][i])
#     dxf_B = np.gradient(f_B)
#     force = -( np.log((h+f_D)/(h+f_D-n_D))*dxf_B +f_B*(1/(h+f_D)*dxf_D-1/(h+f_D-n_D)*(dxf_D-dxn_D)  ) )
#     # plt.plot(force)
#     Force_calc_A[i] = integrate.simpson(f_A*force)
#%% Force calculation


FA = np.zeros(len(t_tasks))
FA_vis = np.zeros(len(t_tasks))
FA_fri = np.zeros(len(t_tasks))
FA_ent = np.zeros(len(t_tasks))
FA_ela = np.zeros(len(t_tasks))
FA_act = np.zeros(len(t_tasks))

FB = np.zeros(len(t_tasks))
FB_vis = np.zeros(len(t_tasks))
FB_fri = np.zeros(len(t_tasks))
FB_ent = np.zeros(len(t_tasks))
FB_ela = np.zeros(len(t_tasks))
FB_act = np.zeros(len(t_tasks))

VA = np.zeros(len(t_tasks))
VB = np.zeros(len(t_tasks))
XA = np.zeros(len(t_tasks))
XB = np.zeros(len(t_tasks))

Overlap = np.zeros(len(t_tasks))
concentration = np.zeros(len(t_tasks))

NA = np.zeros(len(t_tasks))
NB = np.zeros(len(t_tasks))
NAB = np.zeros(len(t_tasks))

# for i in range(len(t_tasks)):
#     FA[i] = tasks['F_A'][i][10]
#     # FA_vis[i] = tasks['F_fA_vis'][i][10]
#     # FA_fri[i] = tasks['F_fA_fri'][i][10]
#     FA_ent[i] = tasks['F_fA_ent'][i][10]
#     FA_ela[i] = tasks['F_fA_ela'][i]
#     # FA_act[i] = tasks['F_fA_act'][i][10]
    
#     FB[i] = tasks['F_B'][i][10]
#     # FB_vis[i] = tasks['F_fB_vis'][i][10]
#     # FB_fri[i] = tasks['F_fB_fri'][i][10]
#     FB_ent[i] = tasks['F_fB_ent'][i][10]
#     FB_ela[i] = tasks['F_fB_ela'][i][10]
#     # FB_act[i] = tasks['F_fB_act'][i][10]
    
#     Overlap[i] = tasks['Overlap'][i][10]
    
#     concentration[i] = tasks['Pab'][i][int(len(x_tasks)/2)]

#     NA[i] = integrate.simpson(tasks['Pa'][i])
#     NB[i] = integrate.simpson(tasks['Pb'][i])
#     NAB[i] = integrate.simpson(tasks['Pab'][i])


FA[i] = tasks['F_A'][i]
# FA_vis[i] = tasks['F_fA_vis'][i]
# FA_fri[i] = tasks['F_fA_fri'][i]
FA_ent[i] = tasks['F_fA_ent'][i]
FA_ela[i] = tasks['F_fA_ela'][i]
# FA_act[i] = tasks['F_fA_act'][i]

FB[i] = tasks['F_B'][i]
# FB_vis[i] = tasks['F_fB_vis'][i]
# FB_fri[i] = tasks['F_fB_fri'][i]
FB_ent[i] = tasks['F_fB_ent'][i]
FB_ela[i] = tasks['F_fB_ela'][i]
# FB_act[i] = tasks['F_fB_act'][i]
for i in range(len(t_tasks)):
    Overlap[i] = tasks['Overlap'][i][10]
    
    concentration[i] = tasks['Pab'][i][int(len(x_tasks)/2)]
    
    NA[i] = integrate.simpson(tasks['Pa'][i])
    NB[i] = integrate.simpson(tasks['Pb'][i])
    NAB[i] = integrate.simpson(tasks['Pab'][i])

#%%
plt.figure(dpi=200)
plt.plot(t_tasks,NA)
plt.plot(t_tasks,NB+1)
plt.plot(t_tasks,NAB)
plt.plot(t_tasks,NA+NB+NAB)

plt.show()

#%% posteriori force calculation

FA_ent_post = np.zeros(len(t_tasks))

for i in range(len(t_tasks)):
    fD = tasks['f_D'][i]
    fA = tasks['f_A'][i]
    fB = tasks['f_B'][i]
    Pab = tasks['Pab'][i]

    dxfD = np.gradient(fD)
    dxfB = np.gradient(fB)
    dxPab = np.gradient(Pab)

    h = 1e-3
    
    # Pab[Pab<h]=h
    # fD[fD<h]=h
    Pab = np.minimum(fD,Pab)
    Pab[Pab<h]=h
    fD[fD<h]=h

    h_1 = 1e-4
    h_2 = 1e-4

    force = -( np.log((h+fD)/(h+fD-Pab))*dxfB +fB*(1/(h+fD)*dxfD-1/(h+fD-Pab)*(dxfD-dxPab)  ) ) 
    FA_ent_post[i] = integrate.simpson(fA*force)


    #problem.add_equation("grad_mu_fA = -1*( np.log((h_log+f_D)/(h_log+f_D-Mab-Pab))*dx(f_B) +f_B*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Mab-Pab)*dx(f_D-Mab-Pab)) + 1/(h+f_A-Mab-Pab)*dx(f_A-Mab-Pab) -1/(h+f_A-Mab-Pab-Ma)*dx(f_A-Mab-Pab-Ma)  )")


#%%
# plt.plot(-FA_ent_post)
# plt.hlines(0.83, 0, 500)
# plt.hlines(1.32, 0, 500)

# plt.show()

#%%
# F_el = Force_el_A
# F_01= Force_ent_A
# F_1000 = Force_calc_A
# F_visc = Force_visc_A


#%%
# F_tot_1s = F_el_1 + F_ent_D1+F_visc_D1

#%% plot the forces
# plt.figure(dpi =200)

# # plt.plot(t_tasks[1:],F_A_tot[1:],label = "tot")
# plt.plot(t_tasks[1:],-8*Force_calc_A[1:])
# plt.plot(t_tasks[1:],-4*1e-3*Force_ent_A[1:],label = r"$f_A$")
# # plt.plot(t_tasks[1:],Force_ent_B[1:],label = r"$f_B$")
# # plt.plot(t_tasks[1:],Force_ent_A[1:]+Force_ent_B[1:],label = r"$\Delta f $")

# # plt.plot(t_tasks[1:],F_A_visc[1:],label = "visc")
# # plt.plot(t_tasks[1:],10000*F_A_el[1:],label = "el")
# # plt.ylim(-5,20)
# plt.legend()

# plt.ylabel("force (10e-3 pN)")
# plt.xlabel("time (s)")
# plt.show()

#%%


#%% Comparaison to the experiment
# Read CSV file into pandas DataFrame
name = "1112_VP_4_jumps_s1.csv"
folder = "velocity_profiles"
path = dir_local

df = pd.read_csv(path +'/'+folder + '/'+name)

# Display the DataFrame (table)
# print(df)

# Put the DataFram in something usable for numpy and me
data = np.transpose(np.array(df))
Force_exp = data[2]
V_exp = data[1]
Force_exp = LS.function_convertion_array(data[2], len(t_tasks))
V_exp = LS.function_convertion_array(data[1], len(t_tasks))


#%% READ RAW TIME AND FORCE DATA
name_raw = "1112_VP_4_jumps_s1_raw.csv"
df_raw = pd.read_csv(path +'/'+folder + '/'+name_raw)
data_raw = np.transpose(np.array(df_raw))
Force_raw = data_raw[2]
T_raw = data_raw[0]
L_raw = data_raw[1]

# Force_raw_conv = LS.function_convertion_array(data_raw[1], len(t_tasks))


#%%#%% FOR FIT

plt.figure(dpi =500)
# plt.plot(t_tasks[1:],Force_exp[1:],label = "exp",color="gray",linewidth=2)

plt.plot(T_raw/1000,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none')
plt.plot(gaussian_filter1d(T_raw/1000, 2),gaussian_filter1d(Force_raw, 2),label = "exp",color="black",linewidth=2)


plt.plot(t_tasks[1:],1*4e-3*tasks["F_fA_ent"][1:],label = r"$F_{ent}$",linewidth=2)




plt.legend(loc="upper left")
plt.ylabel("force (pN)")
plt.xlabel("time (s)")
# plt.axis('off')

# plt.xlim(0,2)
plt.ylim(7,40)
plt.show()





#%%


#%%

plt.figure(dpi =500)

# plt.plot(-Overlap[1:],Force_exp[1:],label = "exp",color="gray",linewidth=2)
# plt.plot(-Overlap[1:],Force_lisse[1:],label = "exp",color="gray",linewidth=2)


plt.plot(L_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none')
plt.plot(gaussian_filter1d(L_raw, 2),gaussian_filter1d(Force_raw, 2),label = "exp",color="black",linewidth=2)
plt.plot(-Overlap[1:],1*4e-3*tasks["F_fA_ent"][1:],label = r"$F_{ent}$",linewidth=2)


# plt.plot(t_tasks[1:],-4e-3*F_01[1:],label = r"$l_i = 0.1$",linewidth=2)
# plt.plot(t_tasks[1:],-4e-3*F_005[1:],label = r"$l_i = 0.05$",linewidth=2)
# plt.plot(t_tasks[1:],-4e-3*F_002[1:],label = r"$l_i = 0.02$",linewidth=2)
# plt.plot(t_tasks[1:],-4e-3*F_001[1:],label = r"$l_i = 0.01$",linewidth=2)

# plt.hlines(11.5, 0,0.8,label=r'$c_i=$'+str(round(concentration[49],4)),color="black")
# plt.hlines(7.2, 0,0.5,label=r'$c_f=$'+str(round(concentration[21],4)),color="black")
# plt.vlines(t_tasks[21], 5,30,color="black")
# plt.vlines(t_tasks[49], 5,30,color="black")



# plt.legend(loc="lower right")
# plt.title("Force with k_off = 0.06/2")
plt.ylabel("force (pN)")
plt.xlabel("overlap ($\mu$m)")
# plt.axis('off')

# plt.xlim(0,0.5)
# plt.ylim(5,20)
plt.show()


#%% Big figure
# name_1 =  "Kucera2021_ch_s1_3"
name_1 =  "Kucera2021_2210_s1_complete_X05"

dir_input_file_1 = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/EntropicForce/'+name_1+'/'
name_input_file_1 = name_1+'_s1'
extension_input_file = '.h5'



#Kucera2021_ch_s1_long2
name_4 =  "Kucera2021_1212_4_fine_s1_k1000_N2000"
name_4_input_file = name_4+'_s1'

name_4 =  "Kucera2021_1212_4_fine_s1_k1000_N2000"
dir_input_file_4 = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/EntropicForce/'+name_4+'/'
name_input_file_4 = name_4+'_s1'
extension_input_file = '.h5'


nameV1 = "2210_VP_1_jump_s1.csv"
nameV4 = "VP_4_jumps_s1.csv"
folderV = "velocity_profiles"






tasks1 = d3.load_tasks_to_xarray(dir_input_file_1+name_input_file_1+extension_input_file) # Downloadig the files
t_tasks1 = np.array(tasks1['f_A']['t'])

tasks4 = d3.load_tasks_to_xarray(dir_input_file_4+name_4_input_file+extension_input_file) # Downloadig the files
t_tasks4 = np.array(tasks4['f_A']['t'])


FA1_ent = np.zeros(len(t_tasks1))
for i in range(len(t_tasks1)):
    FA1_ent[i] = 4e-3*tasks1['F_fA_ent'][i]

FA4_ent = np.zeros(len(t_tasks4))
for i in range(len(t_tasks4)):
    FA4_ent[i] = 4e-3*tasks4['F_fA_ent'][i]



nameV1 = "2210_VP_1_jump_s1_raw.csv"
nameV4 = "1112_VP_4_jumps_s1_raw.csv"
folderV = "velocity_profiles"

df1 = pd.read_csv(path +'/'+folderV + '/'+nameV1)
data1 = np.transpose(np.array(df1))
Force_exp_1 = data1[2]
V_exp_1 = data1[1]

F_exp_1 = LS.function_convertion_array(data1[2], len(t_tasks1))
V_exp_1 = LS.function_convertion_array(data1[1], len(t_tasks1))

df4 = pd.read_csv(path +'/'+folderV + '/'+nameV4)
data4 = np.transpose(np.array(df4))
Force_exp_4 = data4[2]
V_exp_4 = data4[1]

F_exp_4 = LS.function_convertion_array(data4[2], len(t_tasks4))
V_exp_4 = LS.function_convertion_array(data4[1], len(t_tasks4))


#%% RAW force
name_raw = "2210_VP_1_jump_s1_raw.csv"
df_raw = pd.read_csv(path +'/'+folder + '/'+name_raw)
data_raw = np.transpose(np.array(df_raw))
Force_raw = data_raw[2]
T_raw = data_raw[0]
L_raw = data_raw[1]



name_raw_4 = "1112_VP_4_jumps_s1_raw.csv"
df_raw = pd.read_csv(path +'/'+folder + '/'+name_raw_4)
data_raw = np.transpose(np.array(df_raw))
Force_raw_4 = data_raw[2]
T_raw_4 = data_raw[0]
L_raw_4 = data_raw[1]



#%%
# plt.figure(dpi=300,figsize=(6,2))

# plt.plot(Time_raw[1:][200:600],Force_raw[1:][200:600],color="black",linewidth=0,label="experimental data",marker="o",markersize=7, mfc='none')
# # plt.plot(t_tasks1[1:][::10],F_exp_1[1:][::10],color="black",linewidth=0,label="experimental data",marker="o",markersize=7, mfc='none')

# # plt.plot(t_tasks1[-1]+t_tasks4[1:][::5],F_exp_4[1:][::5],color="black",linewidth=0,label="experimental data",marker="o",markersize=7, mfc='none')


# # plt.plot(t_tasks1[1:],FA1_ent[1:],color=color_m[3],linewidth=2,label="fit region")
# # plt.plot(t_tasks1[-1]+t_tasks4[1:],FA4_ent[1:],color=color_m[3],linestyle="dashed",linewidth=2,label="prediction")


# # plt.legend(loc="upper left",fontsize=13)
# # plt.legend(loc="lower right",fontsize=12)

# # plt.title("Force with k_off = 0.06/2")
# plt.tick_params(axis='both', which='major', labelsize=8)
# plt.ylabel("force (pN)",fontsize=8)
# plt.xlabel("time (s)",fontsize=8)

#%%

plt.figure(dpi=1000,figsize=(8*cm,3*cm))

plt.plot(T_raw/1000,Force_raw,color="gray",linewidth=0,marker="o", mfc='none',markersize=3)
plt.plot(gaussian_filter1d(T_raw/1000, 2),gaussian_filter1d(Force_raw, 2),label = "exp smoothed",color="black",linewidth=1)


plt.plot(T_raw[-1]/1000+T_raw_4/1000,Force_raw_4,color="gray",linewidth=0,marker="o", mfc='none',markersize=3)
plt.plot(T_raw[-1]/1000+gaussian_filter1d(T_raw_4/1000, 5),gaussian_filter1d(Force_raw_4, 5),label = "exp smoothed",color="black",linewidth=1)


# plt.plot(t_tasks1[-1]+t_tasks4[1:][::5],F_exp_4[1:][::5],color="black",linewidth=0,label="experimental data",marker="o",markersize=7, mfc='none')


plt.plot(t_tasks1[1:],FA1_ent[1:],color=color_m[1],linewidth=1.5,label="fit region")
plt.plot(t_tasks1[-1]+t_tasks4[1:],FA4_ent[1:],color=color_m[1],linewidth=1.5,label="prediction")

# plt.legend(loc="upper left",fontsize=13)
# plt.legend(loc="lower right",fontsize=12)

# plt.title("Force with k_off = 0.06/2")
plt.tick_params(axis='both', which='major', labelsize=8)
plt.ylabel("force (pN)",fontsize=8)
plt.xlabel("time (s)",fontsize=8)




#%%



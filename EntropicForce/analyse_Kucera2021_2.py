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
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

#%%
dir_local = os.path.dirname(__file__)

name =  "Kucera2021_1112_4_jumps_fit"
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

O_tasks = np.zeros(len(t_tasks))
for i in range(len(t_tasks)):
    O_tasks[i] = np.array(tasks['Overlap'][i][10])




#%% LOAD VELOCITY APPLIED
path_velocity = dir_local
folder_velocity = "velocity_profiles"
name_file_velocity = "1112_VP_4_jumps_s1"
ext = ".csv"

data_pd_velocity = pd.read_csv(path_velocity +'/'+folder_velocity + '/'+name_file_velocity +ext)
print(data_pd_velocity)

data_velocity = np.transpose(np.array(data_pd_velocity))
F_exp = data_velocity[2]
V_exp = data_velocity[1]
F_exp_converted = LS.function_convertion_array(F_exp, len(t_tasks))
V_exp_converted = LS.function_convertion_array(V_exp, len(t_tasks))


name_file_raw = name_file_velocity+"_raw"
data_pd_raw = pd.read_csv(path_velocity +'/'+folder_velocity + '/'+name_file_raw +ext)
data_raw = np.transpose(np.array(data_pd_raw))
Force_raw = data_raw[2]
T_raw = data_raw[0]/1000
L_raw = data_raw[1]

# Force_raw_conv = LS.function_convertion_array(data_raw[1], len(t_tasks))


#%%#%% PLOT OF ONE RESULTS

plt.figure(dpi =200,figsize=(3,2))

plt.plot(T_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none',markersize=5)
plt.plot(gaussian_filter1d(T_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)

plt.plot(t_tasks[1:],1*4e-3*tasks["F_fA_ent"][1:],label = r"$F_{ent}$",linewidth=1.5,color=color_m[1])


plt.tick_params(axis='both', which='major')
plt.ylabel("force (pN)")
plt.xlabel("time (s)")
# plt.axis('off')

# plt.xlim(0,2)
# plt.ylim(5,20)
plt.show()

#%%#%% PLOT - concentration fit

plt.figure(dpi =200,figsize=(3,2))


plt.plot(T_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none',markersize=5)
plt.plot(gaussian_filter1d(T_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)

plt.plot(t_tasks[1:],1*4e-3*tasks["F_fA_ent"][1:],label = r"$F_{ent}$",linewidth=1.5,color=color_m[1])


plt.hlines(27, 0, 30)
plt.hlines(11.5, 0, 30)

index_1 = np.abs(4e-3*tasks["F_fA_ent"] - 11.5).argmin()
index_2 = np.abs(4e-3*tasks["F_fA_ent"] - 27).argmin()

plt.vlines(t_tasks[index_1], 0, 20)
plt.vlines(t_tasks[index_2], 0, 20)

plt.tick_params(axis='both', which='major')
# plt.legend(loc="upper left",fontsize=fontsize_legend_m)
plt.ylabel("force (pN)")
plt.xlabel("time (s)")
# plt.axis('off')
plt.title(r"$C_{start}$="+str(np.array(tasks["Pab"][index_1][int(len(x_tasks)/2)]))+"\n"+
          r"$C_{end}$="+str(np.array(tasks["Pab"][index_2][int(len(x_tasks)/2)])))
# plt.xlim(0,2)
# plt.ylim(5,20)
plt.show()


#%%
DL = L_raw[-1]-L_raw[0]

C_s = np.array(tasks["Pab"][index_1][int(len(x_tasks)/2)])
C_e = np.array(tasks["Pab"][index_2][int(len(x_tasks)/2)])

L_start = -C_e*DL/(C_s-C_e)

print(L_start)
print("xl_start="+str(L_start/2))
print("C_start = "+str(C_s))

#%% PLOT LIST OF RESULTS

extension_input_file = '.h5'

F_list = []
F_list_calc = []



name_file_list= [
    "Kucera2021_2210_s1_complete_X1",
    "Kucera2021_2210_s1_complete_X05",
    "Kucera2021_2210_s1_complete_X025",
    "Kucera2021_2210_s1_complete_X01",
    "Kucera2021_2210_s1_complete_X00",
    ]


# name_file_list= [
#     "Kucera2021_2110_single_X1",
#     "Kucera2021_2110_single_X05",
#     "Kucera2021_2110_single_X025",
#     "Kucera2021_2110_single_X01",
#     "Kucera2021_2110_single_X00",
#     ]




for name_file in name_file_list:
    print(name_file)
    dir_input_file = '//home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/EntropicForce/'+name_file+'/'
    name_input_file = name_file+'_s1'
    tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files
    t_tasks = np.array(tasks['f_A']['t'])
    
    
    F_list.append(4e-3*tasks['F_fA_ent']) # mesure
    
    f_calc = np.ones(len(t_tasks))
    for i in range(len(t_tasks)):
        fD = tasks['f_D'][i]
        fA = tasks['f_A'][i]
        fB = tasks['f_B'][i]
        Pab = tasks['Pab'][i]
        dxfD = np.gradient(fD)
        dxfB = np.gradient(fB)
        dxPab = np.gradient(Pab)
        h = 2e-3
        # Pab[Pab<h]=h
        # fD[fD<h]=h
        Pab = np.minimum(fD,Pab)
        # Pab[Pab<h]=h
        # fD[fD<h]=h
        force = -( np.log((h+fD)/(h+fD-Pab))*dxfB +fB*(1/(h+fD)*dxfD-1/(h+fD-Pab)*(dxfD-dxPab)  ) ) 
        f_calc[i] = integrate.simpson(fA*force)
    F_list_calc.append(f_calc)

#%% single plot for force-time

plt.figure(dpi=500,figsize=(9*cm,8*cm))


plt.plot(T_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none',markersize=5)

for i in range(len(F_list)):
    mask = np.isnan(F_list[i][1:])
    for j in range(len(mask)-2):
        if mask[j] == True:
            mask[j-2] = True
        if mask[j] == True and mask[j+2] == False:
            mask[j+1] = True
            mask[j+2] = True
            break
        
    plt.plot(t_tasks[1:],F_list[i][1:],color=color_m[i],linewidth=1.5,label=name_file_list[i][28:35])
    plt.plot(t_tasks[1:][mask],-16*F_list_calc[i][1:][mask],color=color_m[i],linewidth=1,linestyle="--")



plt.plot(gaussian_filter1d(T_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)






plt.tick_params(axis='both', which='major')
# plt.legend(loc="lower left")
plt.ylabel("force (pN)")
plt.xlabel("time (s)")
# plt.axis('off')

plt.xlim(0.5, 6)
plt.ylim(6, 25  )

# plt.ylim(5,20)
plt.show()


# #%% single plot for force-overlap POSTER

# orange_perso = (255/255, 69/255, 0) 
# bleu_perso = (25/255, 25/255, 112/255) 



# plt.figure(dpi=500,figsize=(9*cm,9*cm))


# plt.plot(T_raw,Force_raw,label = "exp",color=bleu_perso,linewidth=0,marker="o", mfc='none',markersize=5,alpha = 0.2)
# plt.plot(gaussian_filter1d(T_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color=bleu_perso,linewidth=1.5)

# for i in range(len(F_list)):
#     mask = np.isnan(F_list[i][1:])
#     for j in range(len(mask)-2):
#         if mask[j] == True:
#             mask[j-2] = True
#         if mask[j] == True and mask[j+2] == False:
#             mask[j+1] = True
#             mask[j+2] = True
#             break
        
#     plt.plot(t_tasks[1:],F_list[i][1:],color=orange_perso,linewidth=1.5,label=name_file_list[i][28:35])
#     plt.plot(t_tasks[1:][mask],-16*F_list_calc[i][1:][mask],color=color_m[i],linewidth=1,linestyle="--")



# # plt.plot(gaussian_filter1d(T_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)



# plt.tick_params(axis='both', which='major')
# # plt.legend(loc="lower left")

# plt.ylabel("Force (pN)")
# plt.xlabel("Time (min)")
# # plt.axis('off')

# # plt.xlim(-0.015, 0.095)
# plt.ylim(6, 19  )

# # plt.ylim(5,20)
# plt.savefig("poster_ent", dpi=500, bbox_inches="tight", transparent=True)
# plt.show()



#%% single plot for force-overlap

plt.figure(dpi=500,figsize=(4*cm,4.5*cm))


plt.plot(L_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none',markersize=5)

for i in range(len(F_list)):
    mask = np.isnan(F_list[i][1:])
    for j in range(len(mask)-2):
        if mask[j] == True:
            mask[j-2] = True
        if mask[j] == True and mask[j+2] == False:
            mask[j+1] = True
            mask[j+2] = True
            break
        
    plt.plot(-O_tasks[1:],F_list[i][1:],color=color_m[i],linewidth=1.5,label=name_file_list[i][28:35])
    plt.plot(-O_tasks[1:][mask],-16*F_list_calc[i][1:][mask],color=color_m[i],linewidth=1,linestyle="--")



# plt.plot(gaussian_filter1d(T_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)



plt.plot(gaussian_filter1d(L_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)


plt.tick_params(axis='both', which='major')
# plt.legend(loc="lower left")

plt.xlabel("Distance ($\mu$m)")
# plt.axis('off')

plt.xlim(-0.015, 0.095)
plt.ylim(6, 19  )

# plt.ylim(5,20)
plt.show()


#%% single plot for force-overlap ZOOM

plt.figure(dpi=200,figsize=(9*cm,8*cm))


plt.plot(L_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none',markersize=5)

for i in range(len(F_list)):
    plt.plot(-O_tasks[1:],F_list[i][1:],color=color_m[i],linewidth=1.5,label=name_file_list[i][28:35])
    # plt.plot(t_tasks[1:],-16*F_list_calc[i][1:],color=str(0.3+(i)/(len(F_list))*(0.7-0.3)),linewidth=1,label="$n^{(a)}+n^{(b)}+n^{(ab)}$")


plt.plot(gaussian_filter1d(L_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)


plt.tick_params(axis='both', which='major')
# plt.legend(loc="lower left")
plt.ylabel("force (pN)")
plt.xlabel("time (s)")
# plt.axis('off')

plt.xlim(0.06, 0.09)
plt.ylim(10, 16  )

# plt.ylim(5,20)
plt.show()







#%% PLOT LIST OF RESULTS - SAME FOR CHANGE IN KOFF

extension_input_file = '.h5'

F_list = []
F_list_calc = []



name_file_list= [
    "Kucera2021_1411_kx100",
    "Kucera2021_1411_kx300",
    "Kucera2021_1411_kx1000",
    "Kucera2021_1411_kx3000",
    # "Kucera2021_2210_s1_complete_X05",
    # "Kucera2021_1411_test",
    # "Kucera2021_1211_s1_koff_100",
    # "Kucera2021_1211_s1_koff_30",
    
    
    ]



for name_file in name_file_list:
    print(name_file)
    dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/EntropicForce/'+name_file+'/'
    name_input_file = name_file+'_s1'
    tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files
    t_tasks = np.array(tasks['f_A']['t'])
    
    
    F_list.append(4e-3*tasks['F_fA_ent'][:499]) # mesure
    
    f_calc = np.ones(len(t_tasks))
    for i in range(len(t_tasks)):
        fD = tasks['f_D'][i]
        fA = tasks['f_A'][i]
        fB = tasks['f_B'][i]
        Pab = tasks['Pab'][i]
        dxfD = np.gradient(fD)
        dxfB = np.gradient(fB)
        dxPab = np.gradient(Pab)
        h = 2e-3
        # Pab[Pab<h]=h
        # fD[fD<h]=h
        Pab = np.minimum(fD,Pab)
        # Pab[Pab<h]=h
        # fD[fD<h]=h
        force = -( np.log((h+fD)/(h+fD-Pab))*dxfB +fB*(1/(h+fD)*dxfD-1/(h+fD-Pab)*(dxfD-dxPab)  ) ) 
        f_calc[i] = integrate.simpson(fA*force)
    F_list_calc.append(f_calc[:499])

#%% single plot for force-time

plt.figure(dpi=500,figsize=(9*cm,8*cm))


plt.plot(T_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none',markersize=5)

for i in range(len(F_list)):
    mask = np.isnan(F_list[i][:])
    for j in range(len(mask)-2):
        if mask[j] == True:
            mask[j-2] = True
        if mask[j] == True and mask[j+2] == False:
            mask[j+1] = True
            mask[j+2] = True
            break
        
    plt.plot(t_tasks[1:len(F_list[i])],F_list[i][1:],color=color_m[i],linewidth=1.5,label=name_file_list[i][17:])
    plt.plot(t_tasks[1:][mask],-16*F_list_calc[i][:][mask],color=color_m[i],linewidth=1,linestyle="--")



plt.plot(gaussian_filter1d(T_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp smoothed",color="black",linewidth=1.5)





plt.tick_params(axis='both', which='major')
plt.legend(loc="upper right")
plt.ylabel("force (pN)")
plt.xlabel("time (s)")
# plt.axis('off')

plt.xlim(0.5, 6)
plt.ylim(6, 25  )

# plt.ylim(5,20)
plt.show()


#%% single plot for force-overlap

plt.figure(dpi=500,figsize=(9*cm,8*cm))


plt.plot(L_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none',markersize=5)

for i in range(len(F_list)):
    mask = np.isnan(F_list[i][:])
    for j in range(len(mask)-2):
        if mask[j] == True:
            mask[j-2] = True
        if mask[j] == True and mask[j+2] == False:
            mask[j+1] = True
            mask[j+2] = True
            break
        
    plt.plot(-O_tasks[1:],F_list[i],color=color_m[i],linewidth=1.5,label=name_file_list[i][28:35])
    plt.plot(-O_tasks[1:][mask],-16*F_list_calc[i][:][mask],color=color_m[i],linewidth=1,linestyle="--")



# plt.plot(gaussian_filter1d(T_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)



plt.plot(gaussian_filter1d(L_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)


plt.tick_params(axis='both', which='major')
# plt.legend(loc="lower left")

plt.xlabel("Distance ($\mu$m)")
# plt.axis('off')

plt.xlim(-0.015, 0.095)
plt.ylim(6, 19  )

# plt.ylim(5,20)
plt.show()


#%% single plot for force-overlap ZOOM

plt.figure(dpi=200,figsize=(9*cm,8*cm))


plt.plot(L_raw,Force_raw,label = "exp",color="gray",linewidth=0,marker="o", mfc='none',markersize=5)

for i in range(len(F_list)):
    plt.plot(-O_tasks[1:],F_list[i],color=color_m[i],linewidth=1.5,label=name_file_list[i][28:35])
    plt.plot(t_tasks[1:],-16*F_list_calc[i][:],color=str(0.3+(i)/(len(F_list))*(0.7-0.3)),linewidth=1,label="$n^{(a)}+n^{(b)}+n^{(ab)}$")


plt.plot(gaussian_filter1d(L_raw, 1),gaussian_filter1d(Force_raw, 1),label = "exp filter",color="black",linewidth=1.5)


plt.tick_params(axis='both', which='major')
# plt.legend(loc="lower left")
plt.ylabel("force (pN)")
plt.xlabel("time (s)")
# plt.axis('off')

plt.xlim(0.06, 0.09)
plt.ylim(10, 16  )

# plt.ylim(5,20)
plt.show()






#%%
'end'


#%%



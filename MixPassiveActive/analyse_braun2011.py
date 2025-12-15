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
from scipy.optimize import curve_fit
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

name =    "Braun2011_2811_3"
# 
dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/Braun2011/'+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = "output_test1"


bool_anim = 1





#%% loading data
tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files

x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])/60

# x_tasks = np.array(tasks['n_Mab']['x'])
# t_tasks = np.array(tasks['n_Mab']['t'])/60

#%%
n_img = 2
N_end = len(t_tasks)
extension_animation = ".mp4"
frame_per_second = 20


# %% ANIMATION
if bool_anim:
    fig = plt.figure(figsize=(9, 3), dpi=200)
    def animate(i):
        if i%(N_end/100) == 0:
            print(i)
        plt.clf()
        plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$n^{(A)}$")
        plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$n^{(B)}$")
        
        plt.plot(x_tasks,tasks['Pab'][i],color = 'violet',linestyle="-",label = r"$n^{(ab)}_p$")
        plt.plot(x_tasks,tasks['Pa'][i],color = 'blue',linestyle="-",label = r"$n^{a}_p$")
        plt.plot(x_tasks,tasks['Pb'][i],color = 'red',linestyle="-",label = r"$n^{b}_p$")

        plt.plot(x_tasks,tasks['Mab'][i],color = 'violet',linestyle="--",label = r"$n^{ab}_m$")
        plt.plot(x_tasks,tasks['Ma'][i],color = 'blue',linestyle="--",label = r"$n^{a}_m$")
        plt.plot(x_tasks,tasks['Mb'][i],color = 'red',linestyle="--",label = r"$n^{b}_m$")

        # plt.plot(x_tasks,tasks['f_D'][i]-tasks['n_D'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
        plt.gca().set_yticks((0,1))
        plt.gca().set_yticklabels((0,"$n_{max}$"))
        
        # plt.legend(loc='upper left',fontsize=5)
    
        
    t=np.arange(0,N_end,n_img) # New time array with only n images  
    ani = FuncAnimation(fig, animate, frames=t,
                        interval=1, repeat=False)
    #name = "D"+str(D)+"a"+str(alpha)+".gif"
    ani.save(dir_output_file+"/"+name_output_file+"_density"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)




#%% Force calculation



# FA_vis = np.zeros(len(t_tasks))
# FA_fri = np.zeros(len(t_tasks))
# FA_ent = np.zeros(len(t_tasks))
# FA_ela = np.zeros(len(t_tasks))
# FA_act = np.zeros(len(t_tasks))

# FB_vis = np.zeros(len(t_tasks))
# FB_fri = np.zeros(len(t_tasks))
# FB_ent = np.zeros(len(t_tasks))
# FB_ela = np.zeros(len(t_tasks))
# FB_act = np.zeros(len(t_tasks))

# VA = np.zeros(len(t_tasks))
# VB = np.zeros(len(t_tasks))

Overlap = np.zeros(len(t_tasks))

N_Pab = np.zeros(len(t_tasks))
N_O = np.zeros(len(t_tasks))


for i in range(len(t_tasks)):
    # FA_vis[i] = tasks['F_fA_vis'][i][10]
    # FA_fri[i] = tasks['F_fA_fri'][i][10]
    # FA_ent[i] = tasks['F_fA_ent'][i][10]d
    # FA_ela[i] = tasks['F_fA_ela'][i][10]
    # FA_act[i] = tasks['F_fA_act'][i][10]
    
    # FB_vis[i] = tasks['F_fB_vis'][i][10]
    # FB_fri[i] = tasks['F_fB_fri'][i][10]
    # FB_ent[i] = tasks['F_fB_ent'][i][10]
    # FB_ela[i] = tasks['F_fB_ela'][i][10]
    # FB_act[i] = tasks['F_fB_act'][i][10]
    
    N_Pab[i] = integrate.simpson(tasks['Pab'][i],x_tasks)
    N_O[i] = integrate.simpson(tasks['Pab'][i]+tasks['Pa'][i]+tasks['Pb'][i],x_tasks)

    # NB[i] = integrate.simpson(tasks['Pb'][i])
    # NAB[i] = integrate.simpson(tasks['Pab'][i]) 
 
    # VA[i] = tasks['V_A'][i][10]
    # VB[i] = tasks['V_B'][i][10]
    
    Overlap[i] = integrate.simpson(tasks['f_D'][i],x_tasks)

#%%


#%%
for t in range(len(tasks['Pab'])):
    tasks['Pab'][t][tasks['Pab'][t]<=0] = 0
    tasks['Pa'][t][tasks['Pa'][t]<=0] = 0
    tasks['Pb'][t][tasks['Pb'][t]<=0] = 0

    tasks['f_B'][t][tasks['f_B'][t]<=0] = 0
    tasks['f_A'][t][tasks['f_A'][t]<=0] = 0


#%%
A = np.zeros((len(t_tasks),len(x_tasks),3))
# for t in range(len(t_tasks)):
#     print(t)
#     A[t] = tasks['n_D'][t]
#     for i in range(len(x_tasks)):
#         if A[t][i]<=0.021:
#             A[t][i] = -0.1*tasks['f_B'][t][i]

A[:,:,1]= 0+0.85*tasks['Pab']/np.max(tasks['Pab'])     
A[:,:,0]= 0+0.5*tasks['f_B']/np.max(tasks['f_B']) + 0.3*tasks['f_A']/np.max(tasks['f_A'])    

#
#%%
plt.figure(dpi = 300,figsize= (6*cm,12*cm))
# plt.pcolormesh(x_tasks,-t_tasks, tasks['n_D'])
plt.pcolormesh(x_tasks,t_tasks,A)
plt.gca().invert_yaxis()
plt.ylabel("Time (min)")
# plt.gca().get_yaxis().set_visible(False)
plt.xlabel("Distance to template   \n filament end ($\mu$m)    ",fontsize=12)
plt.savefig("Kymograph", dpi=1000, bbox_inches="tight", transparent=True)

plt.show()



#%%
plt.figure(dpi = 300,figsize= (6*cm,12*cm))
plt.plot(4e-3*tasks['F_fB_ent'][1:],t_tasks[1:],color="red" ,label="Entropic")
plt.plot(4e-3*tasks['F_fB_act'][1:],t_tasks[1:],color="green" ,label="Active")
plt.plot(4e-3*tasks['F_fB_fri'][1:],t_tasks[1:],color="blue" ,label="Friction")
# plt.plot(tasks['F_fB_vis'][1:],t_tasks[1:] )
# plt.plot(tasks['F_fB_ent'][1:]+tasks['F_fB_act'][1:]+tasks['F_fB_fri'][1:]+tasks['F_fB_vis'][1:],t_tasks[1:] )
plt.gca().axes.yaxis.set_ticklabels([])
plt.xlabel("Force (pN)")


# plt.gca().set_xticks(np.linspace(-1, 1,3))
plt.gca().set_yticks(np.linspace(0, 20,5))

plt.gca().invert_yaxis()
plt.gca().spines[['right', 'top']].set_visible(False)
plt.gca().spines['left'].set_position('zero')
plt.legend()
plt.savefig("Forces", dpi=500, bbox_inches="tight", transparent=True)

# plt.xlim(-200,2000)
plt.show()



#%%


plt.figure(dpi = 300,figsize= (12*cm,6*cm))
plt.plot(t_tasks[1:],tasks['F_fB_fri'][1:]/tasks['F_fB_vis'][1:] )
plt.ylabel(r'$\gamma^\times/\gamma$')
plt.xlabel("Time (min)")
plt.savefig("Ratio gammas", dpi=500, bbox_inches="tight", transparent=True)
plt.show()


#%%

rho = np.zeros(len(t_tasks))
for i in range(len(t_tasks)):
    rho[i] = integrate.simpson(tasks['Pab'][i],x_tasks)/(integrate.simpson(tasks['f_D'][i],x_tasks))



#%%


fig, ax1 = plt.subplots(dpi = 300,figsize= (12*cm,6*cm))

color = 'red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('V ($\mu$m)', color=color)
ax1.plot(t_tasks[1:],tasks['V_B'][1:],color="red" ,label="V")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'green'
ax2.set_ylabel('$n^{(ab)}_{Ase1}$', color=color)  # we already handled the x-label with ax1
ax2.plot(t_tasks[1:],rho[1:]*200,color="green" ,label="N")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0,0.4*200)
plt.savefig("Velocity and Number", dpi=1000, bbox_inches="tight", transparent=True)
plt.show()



#%%
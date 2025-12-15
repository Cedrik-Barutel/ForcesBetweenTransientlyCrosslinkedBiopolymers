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

name =    "Braun2017_1112"
dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/ActiveTwoFilaments/'+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = "output_test1"


bool_anim = 0

# print(name_save)




#%% loading data
tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files

x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])

# x_tasks = np.array(tasks['n_Mab']['x'])
# t_tasks = np.array(tasks['n_Mab']['t'])/60

#%%
n_img = 5
N_end = len(t_tasks)
extension_animation = ".gif"
frame_per_second = 20


# %% ANIMATION
if bool_anim:
    fig = plt.figure(figsize=(5, 2), dpi=200)
    def animate(i):
        if i%(N_end/100) == 0:
            print(i)
        plt.clf()
        plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$\phi^A$")
        plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
        
        plt.plot(x_tasks,tasks['Pab'][i],color = 'violet',linestyle="-",label = r"$P^{ab}$")
        plt.plot(x_tasks,tasks['Pa'][i],color = 'blue',linestyle="-",label = r"$P^{a}$")
        plt.plot(x_tasks,tasks['Pb'][i],color = 'red',linestyle="-",label = r"$P^{b}$")

        plt.plot(x_tasks,tasks['Mab'][i],color = 'violet',linestyle="--",label = r"$P^{ab}$")
        plt.plot(x_tasks,tasks['Ma'][i],color = 'blue',linestyle="--",label = r"$P^{a}$")
        plt.plot(x_tasks,tasks['Mb'][i],color = 'red',linestyle="--",label = r"$P^{b}$")

        # plt.plot(x_tasks,tasks['f_D'][i]-tasks['n_D'][i], color='red',alpha = 0.5, label = r"$\phi^B$")

        plt.legend(loc='upper left',fontsize=5)
    
        
    t=np.arange(0,N_end,n_img) # New time array with only n images  
    ani = FuncAnimation(fig, animate, frames=t,
                        interval=1, repeat=False)
    #name = "D"+str(D)+"a"+str(alpha)+".gif"
    ani.save(dir_output_file+"/"+name_output_file+"_density"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)





#%%


#%%
for t in range(len(tasks['f_A'])):
    tasks['Mab'][t][tasks['Mab'][t]<=0] = 0
    tasks['f_A'][t][tasks['f_A'][t]<=0] = 0
    tasks['f_B'][t][tasks['f_B'][t]<=0] = 0



#%%
A = np.zeros((len(t_tasks),len(x_tasks),3))
# for t in range(len(t_tasks)):
#     print(t)
#     A[t] = tasks['n_D'][t]
#     for i in range(len(x_tasks)):
#         if A[t][i]<=0.021:
#             A[t][i] = -0.1*tasks['f_B'][t][i]

A[:,:,1]= 0+0.8*tasks['Mab']/np.max(tasks['Mab'])     
A[:,:,0]= 0+0.4*tasks['f_A']/np.max(tasks['f_A']) +0.4*tasks['f_B']/np.max(tasks['f_B'])

#
#%%
plt.figure(dpi = 500,figsize= (8*cm,8*cm))
# plt.pcolormesh(x_tasks,-t_tasks, tasks['n_D'])
plt.pcolormesh(x_tasks-1,t_tasks/60,A)

plt.gca().invert_yaxis()
plt.ylabel("Time (min)")
# plt.gca().get_yaxis().set_visible(False)
plt.xlabel("Distance from the middle ($\mu$m)      ")
# plt.ylim(3*9,0)
plt.gca().set_xticks(np.arange(-5,5,1))
plt.gca().set_yticks(np.arange(0,50,10))

plt.hlines(13, -5, 5,color="white",linestyles="--",linewidth=1)
plt.hlines(35, -5, 5,color="white",linestyles="--",linewidth=1)

plt.xlim(-5,3)
# plt.ylim(120,0)
plt.savefig("poster_ent", dpi=1000, bbox_inches="tight", transparent=True)

plt.show()



#%%
plt.figure(dpi=200,figsize=(8*cm,2*cm))
plt.plot(x_tasks-1,tasks['f_A'][0],color="black",linestyle="--")
plt.plot(x_tasks-1,tasks['f_B'][0],color="black",linestyle="--")

plt.plot(x_tasks-1,tasks['Mab'][0],color="black")
plt.ylabel("Occupancy")
plt.xlabel("Distance from filament end ($\mu$m)   ")
# plt.gca().set_xticks(np.arange(0,6,1))
plt.ylim(-0.05,0.2)

plt.plot()


#%%
plt.figure(dpi=200,figsize=(8*cm,2*cm))
plt.plot(x_tasks-1,tasks['f_A'][100],color="black",linestyle="--")
plt.plot(x_tasks-1,tasks['f_B'][100],color="black",linestyle="--")

plt.plot(x_tasks-1,tasks['Mab'][100],color="black")
plt.ylabel("Occupancy")
plt.xlabel("Distance from filament end ($\mu$m)   ")
# plt.gca().set_xticks(np.arange(0,6,1))
plt.ylim(-0.05,0.2)

plt.plot()


#%%
plt.figure(dpi=200,figsize=(8*cm,2*cm))
plt.plot(x_tasks-1,tasks['f_A'][250],color="black",linestyle="--")
plt.plot(x_tasks-1,tasks['f_B'][250],color="black",linestyle="--")

plt.plot(x_tasks-1,tasks['Mab'][250],color="black")
plt.ylabel("Occupancy")
plt.xlabel("Distance from filament end ($\mu$m)   ")
# plt.gca().set_xticks(np.arange(0,6,1))
plt.ylim(-0.05,0.2)

plt.plot()
#%% TEST
# A = np.zeros((len(t_tasks),len(x_tasks),3))
# # for t in range(len(t_tasks)):
# #     print(t)
# #     A[t] = tasks['n_D'][t]
# #     for i in range(len(x_tasks)):
# #         if A[t][i]<=0.021:
# #             A[t][i] = -0.1*tasks['f_B'][t][i]

# A[:,:,1]= 0+0.85*(tasks['Pab']+tasks['Pa']+tasks['Pb'])/np.max(tasks['Pab']+tasks['Pa']+tasks['Pb'])     
# A[:,:,0]= 0+0.5*tasks['f_B']/np.max(tasks['f_B']) + 0.3*tasks['f_A']/np.max(tasks['f_A'])    

# #
# #%%
# plt.figure(dpi = 200,figsize= (4,6))
# # plt.pcolormesh(x_tasks,-t_tasks, tasks['n_D'])
# plt.pcolormesh(x_tasks,t_tasks,A)
# plt.gca().invert_yaxis()
# plt.ylabel("time (min)")
# plt.xlabel("distance to template filament end ($\mu$m)")
# plt.show()
#%%
# plt.figure(dpi=200,figsize=(6,3))
# plt.rcParams['font.size'] = '14'
# plt.plot(x_tasks,tasks['Pab'][100],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)
# # plt.plot(x_tasks,tasks['n_D'][300],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)
# # plt.plot(x_tasks,tasks['n_D'][650],color=(0.0, 0.6, 0.0),label=r"$n^{(ab)}(t_1)$",linewidth=2.5)

# plt.fill_between(x_tasks,tasks['f_A'][100],color="red",alpha=0.15)
# plt.fill_between(x_tasks,tasks['f_B'][100],color="red",alpha=0.15)

# plt.gca().spines['left'].set_linewidth(2)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_linewidth(2)
# # plt.yticks(color='w')
# # plt.gca().spines['bottom'].set_visible(False)
# # plt.gca().get_xaxis().set_visible(False)
# plt.hlines(0.59,-27,6,color="black",linestyles="--")
# plt.ylim(-0.02,1.02)
# plt.xlim(-22,5.5)

# # plt.legend()
# plt.show()


#%%
plt.figure(dpi = 300,figsize= (8*cm,8*cm))
plt.plot(4e-3*tasks['F_fB_ent'][1:],t_tasks[1:]/60,color="red" ,label="Entropic")
plt.plot(4e-3*tasks['F_fB_act'][1:],t_tasks[1:]/60,color="green" ,label="Active")
plt.plot(4e-3*tasks['F_fB_fri'][1:],t_tasks[1:]/60,color="blue" ,label="Friction")
# plt.plot(4e-3*tasks['F_fB_vis'][1:],t_tasks[1:]/60,color="blue" ,label="Friction")

# plt.plot(4e-3*(tasks['F_fB_ent'][1:]+tasks['F_fB_act'][1:]+tasks['F_fB_fri'][1:]+tasks['F_fB_vis'][1:]),t_tasks[1:],color="blue" ,label="Friction")


plt.gca().axes.yaxis.set_ticklabels([])
plt.xlabel("Force (pN)")


# plt.gca().set_xticks(np.linspace(-1, 1,3))
# plt.gca().set_yticks(np.linspace(0, 2,5))
plt.gca().set_yticks(np.arange(0,50,10))
plt.ylim(0,40)


plt.gca().invert_yaxis()
plt.gca().spines[['right', 'top']].set_visible(False)
plt.gca().spines['left'].set_position('zero')
# plt.legend()
plt.savefig("poster_ent", dpi=1000, bbox_inches="tight", transparent=True)

# plt.xlim(-200,2000)
plt.show()


#%%
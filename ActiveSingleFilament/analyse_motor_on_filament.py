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

name =    "motor_on_filament_1112"
dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/ActiveSingleFilament/'+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = "output_test1"


bool_anim = 1

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

        plt.plot(x_tasks,tasks['Ma'][i],color = 'blue',linestyle="--",label = r"$P^{a}$")

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
    tasks['Ma'][t][tasks['Ma'][t]<=0] = 0
    tasks['f_A'][t][tasks['f_A'][t]<=0] = 0


#%%
A = np.zeros((len(t_tasks),len(x_tasks),3))
# for t in range(len(t_tasks)):
#     print(t)
#     A[t] = tasks['n_D'][t]
#     for i in range(len(x_tasks)):
#         if A[t][i]<=0.021:
#             A[t][i] = -0.1*tasks['f_B'][t][i]

A[:,:,1]= 0+0.8*tasks['Ma']/np.max(tasks['Ma'])     
A[:,:,0]= 0+0.3*tasks['f_A']/np.max(tasks['f_A']) +0.5*tasks['Ma']/np.max(tasks['Ma'])

# whitr band
A[:,:,0]=A[:,:,0]+ 1*LS.function_filament(A[:,:,1], 6.5, 7, x_tasks, 0.01)
A[:,:,1]=A[:,:,1]+ 1*LS.function_filament(A[:,:,1], 6.5, 7, x_tasks, 0.01)
A[:,:,2]=A[:,:,2]+ 1*LS.function_filament(A[:,:,1], 6.5, 7, x_tasks, 0.01)

A[:,:,0][A[:,:,0]>=1] = 1
A[:,:,1][A[:,:,1]>=1] = 1
A[:,:,2][A[:,:,2]>=1] = 1



#
#%%
plt.figure(dpi = 500,figsize= (3,3))
# plt.pcolormesh(x_tasks,-t_tasks, tasks['n_D'])
plt.pcolormesh(x_tasks-1,t_tasks/60,A)
for i in range(6):
    plt.hlines(t_tasks[i*20]/60,-1, 6,linestyle="-",linewidth=1.5,color=color_m[i])
    print(t_tasks[i*20]/60)
plt.hlines(t_tasks[8*20]/60,-1, 6,linestyle="-",linewidth=1.5,color="black")

v = 0.05*(1-0.77) 
r_over_k = 0.77/0.0023

plt.plot(5-x_tasks,(+50-r_over_k*np.log((-x_tasks)/(5-v*r_over_k)+1))/60,color="white",linewidth=2,linestyle="--")

plt.gca().invert_yaxis()
plt.ylabel("Time (min)")
# plt.gca().get_yaxis().set_visible(False)
plt.xlabel("Distance from filament end ($\mu$m)      ")
plt.ylim(3*9,0)
plt.gca().set_xticks(np.arange(0,6,1))

plt.xlim(-0.5,6)
plt.ylim(22,-0.1)

plt.gca().spines[['right', 'top','bottom','left']].set_visible(False)

plt.savefig("poster_ent", dpi=500, bbox_inches="tight", transparent=True)

plt.show()



#%%
plt.figure(dpi = 500,figsize= (3,3))
# plt.pcolormesh(x_tasks,-t_tasks, tasks['n_D'])
plt.pcolormesh(x_tasks-1,t_tasks/60,A)
for i in range(6):
    plt.hlines(t_tasks[i*20]/60,-1, 6,linestyle="-",linewidth=1.5,color=color_m[i])
    print(t_tasks[i*20]/60)
plt.hlines(t_tasks[8*20]/60,-1, 6,linestyle="-",linewidth=1.5,color="black")

v = 0.05*(1-0.77) # 0.01 # 0.005
# r_over_k = 0.9/0.00235 #2
r_over_k = 0.77/0.0023

plt.plot(5-x_tasks,(+50-r_over_k*np.log((-x_tasks)/(5-v*r_over_k)+1))/60,color="white",linewidth=2,linestyle="--")

plt.gca().invert_yaxis()
plt.ylabel("Time (min)")
# plt.gca().get_yaxis().set_visible(False)
plt.xlabel("Distance from filament end ($\mu$m)      ")
plt.ylim(3*9,0)
plt.gca().set_xticks(np.arange(0,6,1))

plt.xlim(-0.5,6)
plt.ylim(22,-0.1)



plt.show()


#%%
print(v*r_over_k)
print(1/r_over_k)

#%%


#%%
t = np.linspace(0,3*8*60,200)

r_b = 0.77
d=0.008
r_max = 1/d
L=5
N=1/d*L
v_0 = 0.05
dt = d/v_0
kon = 0.24/N/dt
rho = r_b*r_max

V_hd = v_0*(1-r_b)




y = (L-V_hd/kon*r_b)*(1-np.exp(-kon*t))


plt.figure(dpi=200)
plt.plot(t/60,y)
# plt.ylim(-0.5,5)
plt.show()


#%%




#%%
plt.figure(dpi=200,figsize=(8*cm,8*cm))
plt.plot(x_tasks-1,tasks['f_A'][0],color="black",linestyle="--")
for i in range(6):
    plt.plot(x_tasks-1,tasks['Ma'][i*20],color=color_m[i])
plt.plot(x_tasks-1,tasks['Ma'][8*20],color="black")
    
plt.ylabel("Occupancy")
plt.xlabel("Distance from filament end ($\mu$m)   ")
plt.gca().set_xticks(np.arange(0,6,1))
plt.xlim(-0.5,5.5)
plt.savefig("distributions", dpi=500, bbox_inches="tight", transparent=True)

plt.plot()





#%%
#%%
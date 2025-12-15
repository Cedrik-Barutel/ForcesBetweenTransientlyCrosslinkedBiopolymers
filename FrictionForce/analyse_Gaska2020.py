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
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib import colormaps
# local library
dir_local = os.path.dirname(__file__)
sys.path.append(dir_local)
lib_path = "/home/cedrik/Documents/filaments-crosslinkers-projects/lib"
sys.path.append(lib_path)
import matplotlib.colors as colors

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



from matplotlib.colors import LinearSegmentedColormap

# Take seismic and sample only the upper half (red side)
orig = plt.cm.seismic
new_colors_red = orig(np.linspace(0.5, 1, 256))  # 0=blue end, 1=red end
new_colors_blue = orig(np.linspace(0, 0.5, 256))  # 0=blue end, 1=red end
new_colors_both = orig(np.linspace(0, 2, 256))  # 0=blue end, 1=red end

red_half = LinearSegmentedColormap.from_list("seismic_red_half", new_colors_red)
blue_half = LinearSegmentedColormap.from_list("seismic_red_blue", new_colors_blue)
both = LinearSegmentedColormap.from_list("seismic_red_both", new_colors_both)



#%%
dir_local = os.path.dirname(__file__)

name =    "Gaska2020_2409_run_autoV0100N50"
dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/FrictionForce/'+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = "output_test1"


bool_anim = 1

# print(name_save)




#%% loading data
tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files
# Vname_list = ['0025']

x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])

# x_tasks = np.array(tasks['n_Mab']['x'])
# t_tasks = np.array(tasks['n_Mab']['t'])/60

#%%
n_img = 2
N_end = len(t_tasks)
extension_animation = ".mp4"
frame_per_second = 20


# %% ANIMATION
if bool_anim:
    fig = plt.figure(figsize=(9, 3), dpi=500)
    def animate(i):
        if i%(N_end/100) == 0:
            print(i)
        plt.clf()
        plt.plot(x_tasks,tasks['f_A'][i], color='blue',alpha = 0.5, label = r"$n^{(A)}$")
        plt.plot(x_tasks,tasks['f_B'][i], color='red',alpha = 0.5, label = r"$n^{(B)}$")
        
        plt.plot(x_tasks,tasks['Pab'][i],color = 'violet',linestyle="-",label = r"$n^{(ab)}$")
        plt.plot(x_tasks,tasks['Pa'][i],color = 'blue',linestyle="-",label = r"n^{(a)}$")
        plt.plot(x_tasks,tasks['Pb'][i],color = 'red',linestyle="-",label = r"$n^{(b)}$")

        # plt.plot(x_tasks,tasks['Mab'][i],color = 'violet',linestyle="--",label = r"$P^{ab}$")
        # plt.plot(x_tasks,tasks['Ma'][i],color = 'blue',linestyle="--",label = r"$P^{a}$")
        # plt.plot(x_tasks,tasks['Mb'][i],color = 'red',linestyle="--",label = r"$P^{b}$")

        # plt.plot(x_tasks,tasks['f_D'][i]-tasks['n_D'][i], color='red',alpha = 0.5, label = r"$\phi^B$")
        plt.gca().set_yticks((0,1))
        plt.gca().set_yticklabels((0,"$n_{max}$"))
    
        
    t=np.arange(0,N_end,n_img) # New time array with only n images  
    ani = FuncAnimation(fig, animate, frames=t,
                        interval=1, repeat=False)
    #name = "D"+str(D)+"a"+str(alpha)+".gif"
    ani.save(dir_output_file+"/"+name_output_file+"_density"+extension_animation, writer = 'ffmpeg', fps = frame_per_second)


#%% Force calculation


Overlap = np.zeros(len(t_tasks))
N_Pab = np.zeros(len(t_tasks))
N_O = np.zeros(len(t_tasks))
N_s = np.zeros(len(t_tasks))


for i in range(len(t_tasks)):
    
    N_Pab[i] = integrate.simpson(tasks['Pab'][i],x_tasks)
    N_O[i] = integrate.simpson(tasks['Pab'][i]+(tasks['Pa'][i]+tasks['Pb'][i])*tasks['f_D'][i],x_tasks)
    N_s[i] = integrate.simpson((tasks['Pa'][i]+tasks['Pb'][i])*tasks['f_D'][i],x_tasks)

    Overlap[i] = integrate.simpson(tasks['f_D'][i],x_tasks)

#%%





#%%
ib = 1
ie = 420
#%%
# plt.figure(dpi=200)
# plt.title("$V_B(t)$")
# plt.plot(t_tasks[ib:ie],tasks['V_B'][ib:ie],label = r"$V_B(t)$")
# plt.plot(t_tasks[ib:ie],Overlap[ib:ie],label = r"$Overlap$")
# plt.legend()
# plt.show()

A = np.ones(10)

#%%
tasks['F_B'] = tasks['F_B'][~np.isnan(tasks['F_B'])]
print(tasks['F_B'])

#%%
plt.figure(dpi=200)
plt.title("$f(t)$")
plt.plot(t_tasks[ib:ie],4e-3*(tasks['F_B'][ib:ie]-2*tasks['F_fB_ela'][ib:ie]),label = r"$F_B(t)$")
plt.plot(t_tasks[ib:ie],4e-3*tasks['F_fB_ent'][ib:ie],label = r"ent")
plt.plot(t_tasks[ib:ie],4e-3*tasks['F_fB_fri'][ib:ie],label = r"fri")
plt.plot(t_tasks[ib:ie],4e-3*tasks['F_fB_vis'][ib:ie],label = r"vis")
plt.plot(t_tasks[ib:ie],-4e-3*tasks['F_fB_ela'][ib:ie],label = r"ela")

# plt.plot(t_tasks,Overlap,label = r"$Overlap$")
plt.legend()
plt.show()


#%%
F_fri_mean = 4e-3*np.mean(tasks['F_fB_fri'])
F_vis_mean = 4e-3*np.mean(tasks['F_fB_vis'])

#%%


#%%
plt.figure(dpi=200)
plt.plot(t_tasks[ib:ie],200*N_Pab[ib:ie],label="N_Pab")
plt.plot(t_tasks[ib:ie],200*N_O[ib:ie],label = "N_O")
plt.plot(t_tasks[ib:ie],200*N_s[ib:ie],label = "N_s")

plt.legend()
plt.show()

#%%

plt.figure(dpi=200)
plt.title("Normalized F/O")
plt.plot(Overlap[ib:ie],-4e-3*tasks['F_B'][ib:ie]/np.mean(-4e-3*tasks['F_B'][ib:ie]),label="force")
plt.legend()
plt.ylim(0,2)
plt.show()


#%%
plt.figure(dpi=200)
plt.title("Intensity/O")
plt.plot(Overlap[ib:ie],N_Pab[ib:ie]/np.max(N_Pab[ib:ie]),label="Pab")
plt.plot(Overlap[ib:ie],N_O[ib:ie]/np.max(N_O[ib:ie]),label="tot_O")

plt.legend()
plt.ylim(-0.1,1.1)
plt.show()

#%%
plt.figure(dpi=200)
plt.title("Density/O")
plt.plot(Overlap[ib:ie],N_Pab[ib:ie]/(Overlap[ib:ie])/(N_Pab[ib]/Overlap[ib]),label="Pab")
plt.plot(Overlap[ib:ie],N_O[ib:ie]/(Overlap[ib:ie])/(N_O[ib]/Overlap[ib]),label="tot_O")
plt.ylim(-0.1,15)
plt.legend()
plt.show()


#%% FIGURES

FB_tot = -4*1e-3*tasks['F_B'][~np.isnan(tasks['F_B'])][1:]
FB_ent = -4*1e-3*tasks['F_fB_ent'][~np.isnan(tasks['F_fB_ent'])][1:]
FB_fri = -4*1e-3*tasks['F_fB_fri'][~np.isnan(tasks['F_fB_fri'])][1:]




plt.figure(dpi=500,figsize=(8*cm,8*cm))
# plt.hlines(1, 0, 5,color="gray",linewidth = 2,linestyles="--",label="1")
plt.plot(Overlap[:len(FB_tot)],FB_tot/np.mean(FB_tot),color="black",linewidth = 2,label = "Total")
plt.plot(Overlap[:len(FB_ent)],FB_ent/np.mean(FB_tot),color="red",linewidth = 2,label = "Entropic")
plt.plot(Overlap[:len(FB_fri)],FB_fri/np.mean(FB_tot),color="blue",linewidth = 2,label = "Friction")

# plt.ylim(0.5,1.5)
plt.xlim(-0.1,5.3)

plt.ylabel("Normalized Force")
plt.xlabel("Overlap Length ($\mu$m)")
plt.legend()
plt.savefig("plot_output_curves.png", dpi=500, bbox_inches="tight", transparent=True)

plt.show()


#%% writing all the forces

F_list = []
Farray_list = []

O_list = []

V_list = [0.025,0.050,0.075,0.100,0.150,0.200]
N_list = [5,10,20,30,40,50]

Vname_list = ['0025','0050','0075','0100','0150','0200']
Nname_list = ['5','10','20','30','40','50']




index_N = 0
index_V = 0






for index_N in range(len(N_list)):
    F_list.append([])
    O_list.append([])
    Farray_list.append([])

    for index_V in range(len(V_list)):

        print(V_list[index_V])
        print(N_list[index_N])
        
        name = "Gaska2020_1011_MAP_run_autoV"+str(Vname_list[index_V])+"N"+str(Nname_list[index_N]) # name of the simulation 

        # name = "Gaska2020_2409_run_autoV"+str(Vname_list[index_V])+"N"+str(Nname_list[index_N]) # name of the simulation 
        dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/FrictionForce/'+name+'/'
        name_input_file = name+'_s1'
        extension_input_file = '.h5'
    
        tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files
        x_tasks = np.array(tasks['f_A']['x'])
        t_tasks = np.array(tasks['f_A']['t'])
        
        # plt.figure(dpi=100)
        # plt.title("N="+str(N_list[index_N])+" V="+str(V_list[index_V]))
        # plt.plot(t_tasks[ib:ie],4e-3*tasks['F_B'][ib:ie],label = r"$F_B(t)$")
        # plt.plot(t_tasks[ib:ie],4e-3*tasks['F_fB_ent'][ib:ie],label = r"ent")
        # plt.plot(t_tasks[ib:ie],4e-3*tasks['F_fB_fri'][ib:ie],label = r"fri")
        # plt.plot(t_tasks[ib:ie],4e-3*tasks['F_fB_vis'][ib:ie],label = r"vis")
        # plt.legend()
        # plt.show()

        
        F_sim = np.array(-4e-3*tasks['F_B'])
        F_sim = F_sim[~np.isnan(F_sim)]
        F = np.mean(F_sim)
        print(F)
        F_list[index_N].append(F)
        F_sim = F_sim/np.mean(F_sim)
        Overlap = np.zeros(len(t_tasks))
        N_Pab = np.zeros(len(t_tasks))
        # N_O = np.zeros(len(t_tasks))
        # N_s = np.zeros(len(t_tasks))
        for i in range(len(t_tasks)):        
            # N_Pab[i] = integrate.simpson(tasks['Pab'][i],x_tasks)
            # N_O[i] = integrate.simpson(tasks['Pab'][i]+(tasks['Pa'][i]+tasks['Pb'][i])*tasks['f_D'][i],x_tasks)
            # N_s[i] = integrate.simpson((tasks['Pa'][i]+tasks['Pb'][i])*tasks['f_D'][i],x_tasks)
            Overlap[i] = integrate.simpson(tasks['f_D'][i],x_tasks)
        O_list[index_N].append(Overlap)
        Farray_list[index_N].append(F_sim)
        
        # FB_tot = -4*1e-3*tasks['F_B'][~np.isnan(tasks['F_B'])][1:]
        # FB_ent = -4*1e-3*tasks['F_fB_ent'][~np.isnan(tasks['F_fB_ent'])][1:]
        # FB_fri = -4*1e-3*tasks['F_fB_fri'][~np.isnan(tasks['F_fB_fri'])][1:]
        # plt.plot(Overlap[:len(FB_tot)],FB_tot/np.mean(FB_tot),color="black",linewidth = 2,label = "Total")
        # plt.plot(Overlap[:len(FB_ent)],FB_ent/np.mean(FB_tot),color="blue",linewidth = 2,label = "Entropic")
        # plt.plot(Overlap[:len(FB_fri)],FB_fri/np.mean(FB_tot),color="red",linewidth = 2,label = "Friction")
        
  
# plt.ylim(0,2)        
# plt.show()


#%%

plt.figure()
for index_N in range(len(N_list)):
    for index_V in range(len(V_list)):
        plt.plot(O_list[index_N][index_V][1:len(Farray_list[index_N][index_V])],Farray_list[index_N][index_V][1:])    
    
plt.ylim(0.8,1.2)    
plt.show()
    
#%%
plt.figure(dpi=500,figsize=(3,3))
plt.scatter(V_list,F_list[0],label = "_nolegend_",s=10)
plt.plot(V_list,F_list[0],label = N_list[0])

plt.scatter(V_list,F_list[1],label = "_nolegend_",s=10)
plt.plot(V_list,F_list[1],label = N_list[1])

plt.scatter(V_list,F_list[2],label = "_nolegend_",s=10)
plt.plot(V_list,F_list[2],label = N_list[2])

plt.scatter(V_list,F_list[3],label = "_nolegend_",s=10)
plt.plot(V_list,F_list[3],label = N_list[3])

plt.scatter(V_list,F_list[4],label = "_nolegend_",s=10)
plt.plot(V_list,F_list[4],label = N_list[4])

plt.scatter(V_list,F_list[5],label = "_nolegend_",s=10)
plt.plot(V_list,F_list[5],label = N_list[5])
# plt.plot(V_list,(F_list[5][-1]-F_list[5][0])/(V_list[-1]-V_list[0])*np.array(V_list),label = N_list[5])


plt.ylabel("Resisting Force (pN)")

plt.xlabel("Sliding Velocity (nm.s$^{-1}$)")
plt.legend()

plt.show()







#%%

#%% writing all the forces

F_list = []



Dam2_list = [0,0.01,0.05,0.1,0.5,1,10,30]
c_list = [0.005,0.01,0.02,0.05,0.1,0.2]
Dam2name_list = ['0','001','005','01','05','1','10','30']
cname_list = ['0005','001','002','005','01','02']


# Dam2_list = [0.05]
# c_list = [0.05]
# Dam2name_list = ['005']
# cname_list = ['005']


index_Dam2 = 0
index_c = 0


derivees_map = np.zeros((len(Dam2_list),len(c_list)))
test_map = np.zeros((len(Dam2_list),len(c_list)))

MAP_ent_norm = np.zeros((len(Dam2_list),len(c_list)))
MAP_fri_norm = np.zeros((len(Dam2_list),len(c_list)))
MAP_vis_norm = np.zeros((len(Dam2_list),len(c_list)))
MAP_tot_norm = np.zeros((len(Dam2_list),len(c_list)))


for index_Dam2 in range(len(Dam2_list)):
    F_list.append([])
    for index_c in range(len(c_list)):
        
        print(Dam2_list[index_Dam2])
        print(c_list[index_c])
        name = "Gaska2020_0411_test_run_auto_Dam2"+str(Dam2name_list[index_Dam2])+"c"+str(cname_list[index_c]) # name of the simulation 
        dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/FrictionForce/'+name+'/'
        name_input_file = name+'_s1'
        extension_input_file = '.h5'
    
        tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files
        x_tasks = np.array(tasks['f_A']['x'])
        t_tasks = np.array(tasks['f_A']['t'])
        

        F_sim = np.array(-4e-3*tasks['F_B'])
        F_sim = F_sim[~np.isnan(F_sim)]
        F = np.mean(F_sim)
        
        
        FB = -4*1e-3*tasks['F_B'][~np.isnan(tasks['F_B'])][1:-5]
        FB_ent = -4*1e-3*tasks['F_fB_ent'][~np.isnan(tasks['F_fB_ent'])][1:-5]
        FB_fri = -4*1e-3*tasks['F_fB_fri'][~np.isnan(tasks['F_fB_fri'])][1:-5]
        FB_vis = -4*1e-3*tasks['F_fB_vis'][~np.isnan(tasks['F_fB_vis'])][1:-5]


        len_min = np.min((len(FB),len(FB_ent),len(FB_fri)))
        
        y_d = 455 # disconnection index
         
        # i_d = np.min(y_d,len_min)
        # FB = FB/np.mean(abs(FB))
        # FB_ent = FB_ent/np.mean(abs(FB_ent))
        # FB_fri = FB_fri/np.mean(abs(FB_fri))
        # FB_vis = FB_vis/np.mean(abs(FB_vis))
        
        if y_d < len_min:
            i_d = np.where(FB_ent == np.max(FB_ent[:y_d]))[0][0]
            FB = FB[:i_d+1]
            
            # FB_ent = FB_ent/np.mean(abs(FB_ent))
            # FB_fri = FB_fri/np.mean(abs(FB_fri))
            # FB_vis = FB_vis/np.mean(abs(FB_vis))
            
            MAP_tot_norm[index_Dam2][index_c] = (FB[i_d]-FB[0])/FB[0] 
            MAP_ent_norm[index_Dam2][index_c] = (FB_ent[i_d]-FB_ent[0])/FB[0]
            MAP_fri_norm[index_Dam2][index_c] = (FB_fri[i_d]-FB_fri[0])/FB[0] 
            
        if y_d >= len_min:
            MAP_tot_norm[index_Dam2][index_c] = (FB[-1]-FB[0])/FB[0]
            MAP_ent_norm[index_Dam2][index_c] = (FB_ent[-1]-FB_ent[0])/FB[0]
            MAP_fri_norm[index_Dam2][index_c] = (FB_fri[-1]-FB_fri[0])/FB[0]  
        
        
        
        # FB = FB[1:100]
        # FB = FB/np.mean(abs(FB))
        # FB_ent = FB_ent/np.mean(abs(FB_ent))
        # FB_fri = FB_fri/np.mean(abs(FB_fri))
        # FB_vis = FB_vis/np.mean(abs(FB_vis))

        # Df_tot = np.gradient(FB[:len_min],t_tasks[:len_min])
        # Df_ent = np.gradient(FB_ent[:len_min],t_tasks[:len_min])
        # Df_fri = np.gradient(FB_fri[:len_min],t_tasks[:len_min])      
        # Df_vis = np.gradient(FB_vis[:len_min],t_tasks[:len_min])      

        # MAP_ent_norm[index_Dam2][index_c] = integrate.simpson(Df_ent)/len_min#/integrate.simpson(Df_tot)
        # MAP_fri_norm[index_Dam2][index_c] = integrate.simpson(Df_fri)/len_min#/integrate.simpson(Df_tot)
        # MAP_vis_norm[index_Dam2][index_c] = integrate.simpson(Df_vis)/len_min#/integrate.simpson(Df_tot)
        # MAP_tot_norm[index_Dam2][index_c] = integrate.simpson(Df_tot)/len_min#/integrate.simpson(Df_tot)

        # print()
        

            
        # plt.figure()
        # plt.title(name)
        # plt.plot(FB)
        # plt.plot(FB_ent)
        # plt.plot(FB_fri)
        # # plt.plot(FB_vis)
        # plt.vlines(y_d,0, 1)
        # plt.vlines(i_d,0, 1)

        # plt.ylim(-2,2)
        # plt.xlim(-20,520)

        # plt.show()
        
        # print(FB[-1]-FB[0])
        
        # print(integrate.simpson(FB))


#%%

M = 1/0.05
Dam2_list_M = [i * M for i in Dam2_list]

y_ax = np.linspace(0.5,7.5,8)
x_ax = np.linspace(0.5,5.5,6)

plt.figure(dpi=500,figsize=(8*cm,8*cm))
plt.pcolormesh(MAP_ent_norm,edgecolors='w', linewidths=0.0,cmap=red_half)

cbar = plt.colorbar()
# cbar.set_label(" %",rotation=0)


plt.gca().set_yticks(y_ax)
# plt.gca().set_xticks(x_ax)
plt.gca().axes.xaxis.set_ticklabels([])
plt.ylabel('$\mathcal{Dam}/\mathcal{Dam}_{exp}$')
plt.xlabel("Density")
plt.gca().set_yticklabels(Dam2_list_M)
# plt.gca().set_xticklabels(c_list,rotation = 90,visible="False")
# ax.xaxis.set_visible(False)
# plt.gca().set_xtickvisible("False")

plt.savefig("plot_output_MAP_ent.png", dpi=500, bbox_inches="tight", transparent=True)

plt.show()

#%%
y_ax = np.linspace(0.5,7.5,8)
x_ax = np.linspace(0.5,5.5,6)

plt.figure(dpi=500,figsize=(8*cm,7*cm))
plt.pcolormesh(MAP_fri_norm,edgecolors='w', linewidths=0.0,cmap=blue_half)
plt.gca().set_yticks(y_ax)
plt.gca().set_xticks(x_ax)
cbar = plt.colorbar()
# cbar.set_label(" %",rotation=0)
plt.ylabel('$\mathcal{Dam}/\mathcal{Dam}_{exp}$')
plt.xlabel("Density")
plt.gca().set_yticklabels(Dam2_list_M)
plt.gca().set_xticklabels(c_list,rotation = 90)
plt.savefig("plot_output_MAP_fri.png", dpi=500, bbox_inches="tight", transparent=True)

plt.show()


# #%%
# plt.figure(dpi=500,figsize=(12*cm,8*cm))
# plt.pcolormesh(MAP_ent_norm+MAP_fri_norm,edgecolors='w', linewidths=0.0,cmap="bwr_r",vmin=-0.1,vmax=0.1)
# plt.colorbar()
# plt.gca().set_yticks(np.linspace(0.5,7.5,8))
# plt.gca().set_xticks(np.linspace(0.5,5.5,6))
# plt.ylabel('$\mathcal{Da}$')
# plt.xlabel("Density")
# plt.gca().set_yticklabels(Dam2_list)
# plt.gca().set_xticklabels(c_list)

# plt.show()

#%%

new_colors_both = orig(np.linspace(0, 1., 256))  # 0=blue end, 1=red end
both = LinearSegmentedColormap.from_list("seismic_red_both", new_colors_both)


plt.figure(dpi=500,figsize=(8*cm,8*cm))
plt.pcolormesh(MAP_tot_norm,edgecolors='w', linewidths=0.0,cmap=both,vmin=-1,vmax=1)
plt.colorbar()
plt.scatter(3.5,2.5,marker="x",color="black",s=50,linewidths=4)
plt.scatter(1.5,1.5,marker="x",color="black",s=50,linewidths=4)
plt.scatter(3.5,4.5,marker="x",color="black",s=50,linewidths=4)

plt.gca().set_yticks(np.linspace(0.5,7.5,8))
plt.gca().set_xticks(np.linspace(0.5,5.5,6))
# plt.gca().axes.yaxis.set_ticklabels([])

plt.ylabel('$\mathcal{Dam}/\mathcal{Dam}_{exp}$')
plt.xlabel("Density")
plt.gca().set_yticklabels(Dam2_list_M)
plt.gca().set_xticklabels(c_list,rotation = 90)
# plt.hlines(2.5,0, 6,linestyles="--",color="black")


# xc, yc = 4, 2.5     # center
# w,  h  = 3.3, 1.1   # width, height
# angle = 0       # rotation in degrees

# ellipse = patches.Rectange(
#     (xc, yc),
#     w,
#     h,
#     angle=angle,
#     fill=False,       # outline only
#     linewidth=2,
#     edgecolor='black',
#     # facecolor='red',
#     # edgecolor='black',
#     # alpha=0.5  
# )
# plt.gca().add_patch(ellipse)

# rect = patches.Rectangle(
#     (0, 2.25),      # (x, y) of lower-left corner
#     6,             # width
#     0.5,             # height
#     angle=angle,   # rotation (available since Matplotlib 3.4)
#     fill=False,    # outline only
#     linewidth=2,
#     edgecolor='black'
# )

# plt.gca().add_patch(rect)

plt.savefig("plot_output_MAP_tot.png", dpi=500, bbox_inches="tight", transparent=True)

plt.show()











#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
#%%


red_half = colormaps["seismic"].resampled(256).copy().with_extents(0.5, 1.0)

fig, ax = plt.subplots(figsize=(1, 6), layout='constrained')

cmap = red_half
norm = mpl.colors.Normalize(vmin=-1, vmax=1)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='vertical', label='Some Units')


#%%


# Example usage
plt.imshow(MAP_ent_norm/np.max(np.abs(MAP_tot_norm)), cmap=red_half)

plt.colorbar()
plt.show()


#%%



#%%

#%% writing all the forces

F_list = []
O_list = []



# V_list = [0.025,0.050,0.075,0.100,0.150,0.200]
# N_list = [5,10,20,30,40,50]

# Vname_list = ['0025','0050','0075','0100','0150','0200']
# Nname_list = ['5','10','20','30','40','50']



# V_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.30000000000000004]
# N_list = [  1.,  26.,  51.,  76., 101., 126., 151., 176., 201.]


V_list = [ 0.003, 0.01, 0.03, 0.1, 0.30000000000000004]
N_list = [26.,  51.,  76., 101., 126., 151., 176., 201.]



# Vname_list = ['0001', '0003', '001', '003', '01', '030000000000000004', '1']

# Nname_list =['10', '260', '510', '760', '1010', '1260', '1510', '1760', '2010']


Vname_list = ['0003', '001', '003', '01', '030000000000000004']
Nname_list =[ '260', '510', '760', '1010', '1260', '1510', '1760', '2010']

index_N = 0
index_V = 0


derivees_map = np.zeros((len(N_list),len(V_list)))
test_map = np.zeros((len(N_list),len(V_list)))

MAP_ent_norm = np.zeros((len(N_list),len(V_list)))
MAP_fri_norm = np.zeros((len(N_list),len(V_list)))
MAP_vis_norm = np.zeros((len(N_list),len(V_list)))
MAP_tot_norm = np.zeros((len(N_list),len(V_list)))

# plt.figure(dpi=100)

for index_N in range(len(N_list)):
    F_list.append([])
    O_list.append([])
    Farray_list.append([])

    for index_V in range(len(V_list)):

        print("V="+str(V_list[index_V]))
        print("N="+str(N_list[index_N]))
        name = "Gaska2020_1111_MAP_run_autoV"+str(Vname_list[index_V])+"N"+str(Nname_list[index_N]) # name of the simulation 
        dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/ForcesBetweenTransientlyCrosslinkedBiopolymers/FrictionForce/'+name+'/'
        name_input_file = name+'_s1'
        extension_input_file = '.h5'
    
        tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files
        x_tasks = np.array(tasks['f_A']['x'])
        t_tasks = np.array(tasks['f_A']['t'])
        

        F_sim = np.array(-4e-3*tasks['F_B'])
        F_sim = F_sim[~np.isnan(F_sim)]
        F = np.mean(F_sim)
        
        
        FB = -4*1e-3*tasks['F_B'][~np.isnan(tasks['F_B'])][1:-5]
        FB_ent = -4*1e-3*tasks['F_fB_ent'][~np.isnan(tasks['F_fB_ent'])][1:-5]
        FB_fri = -4*1e-3*tasks['F_fB_fri'][~np.isnan(tasks['F_fB_fri'])][1:-5]
        FB_vis = -4*1e-3*tasks['F_fB_vis'][~np.isnan(tasks['F_fB_vis'])][1:-5]


        len_min = np.min((len(FB),len(FB_ent),len(FB_fri)))
        
        y_d = 500 # disconnection index
         
        # i_d = np.min(y_d,len_min)
        # FB = FB/np.mean(abs(FB))
        # FB_ent = FB_ent/np.mean(abs(FB_ent))
        # FB_fri = FB_fri/np.mean(abs(FB_fri))
        # FB_vis = FB_vis/np.mean(abs(FB_vis))
        
        if y_d < len_min:
            i_d = np.where(FB_ent == np.max(FB_ent[:y_d]))[0][0]-5
            FB = FB[:i_d+1]
            
            # FB_ent = FB_ent/np.mean(abs(FB_ent))
            # FB_fri = FB_fri/np.mean(abs(FB_fri))
            # FB_vis = FB_vis/np.mean(abs(FB_vis))
            
            MAP_tot_norm[index_N][index_V] = np.array(FB[i_d]-FB[0])/FB[0] 
            MAP_ent_norm[index_N][index_V] = np.array(FB_ent[i_d]-FB_ent[0])/FB[0]
            MAP_fri_norm[index_N][index_V] = np.array(FB_fri[i_d]-FB_fri[0])/FB[0] 
            
        if y_d >= len_min:
            # print("yes")
            MAP_tot_norm[index_N][index_V] = np.array(FB[-1]-FB[0])/FB[0]
            MAP_ent_norm[index_N][index_V] = np.array(FB_ent[-1]-FB_ent[0])/FB[0]
            MAP_fri_norm[index_N][index_V] = np.array(FB_fri[-1]-FB_fri[0])/FB[0]  
        
        
        
        # plt.figure()
        # plt.title("V="+str(V_list[index_V])+" N="+str(N_list[index_N])+"DF="+str( np.array((FB[-1]-FB[0])/FB[0]))) 
        # plt.plot(FB)
        # plt.plot(FB_ent)
        # plt.plot(FB_fri)
        # # plt.plot(FB_vis)
        # plt.vlines(y_d,0, 1)
        # plt.vlines(i_d,0, 1)

        # # plt.ylim(-2,2)
        # # plt.xlim(-20,520)

        # plt.show()
        print(MAP_tot_norm[index_N][index_V])

 #%%
y_ax = np.linspace(0.5,len(V_list)-0.5,len(V_list))
x_ax = np.linspace(0.5,len(N_list)-0.5,len(N_list))

plt.figure(dpi=500,figsize=(8*cm,8*cm))
plt.pcolormesh(np.transpose(MAP_ent_norm),edgecolors='w', linewidths=0.0,cmap=red_half)

plt.colorbar()

plt.gca().set_yticks(y_ax)
plt.gca().set_xticks(x_ax)
plt.ylabel('V')
plt.xlabel("N")
plt.gca().set_yticklabels(V_list)
plt.gca().set_xticklabels(N_list,rotation = 90)
plt.show()



#%%
y_ax = np.linspace(0.5,len(V_list)-0.5,len(V_list))
x_ax = np.linspace(0.5,len(N_list)-0.5,len(N_list))

plt.figure(dpi=500,figsize=(8*cm,8*cm))
plt.pcolormesh(np.transpose(MAP_fri_norm),edgecolors='w', linewidths=0.0,cmap=blue_half)
plt.gca().set_yticks(y_ax)
plt.gca().set_xticks(x_ax)
plt.ylabel('V')
plt.xlabel("N")
plt.gca().set_yticklabels(V_list)
plt.gca().set_xticklabels(N_list,rotation = 90)
plt.colorbar()
plt.show()


# #%%
# plt.figure(dpi=500,figsize=(12*cm,8*cm))
# plt.pcolormesh(MAP_ent_norm+MAP_fri_norm,edgecolors='w', linewidths=0.0,cmap="bwr_r",vmin=-0.1,vmax=0.1)
# plt.colorbar()
# plt.gca().set_yticks(np.linspace(0.5,7.5,8))
# plt.gca().set_xticks(np.linspace(0.5,5.5,6))
# plt.ylabel('$\mathcal{Da}$')
# plt.xlabel("Density")
# plt.gca().set_yticklabels(Dam2_list)
# plt.gca().set_xticklabels(c_list)

# plt.show()

#%%

# new_colors_both = orig(np.linspace(1., 0., 256))  # 0=blue end, 1=red end
# both = LinearSegmentedColormap.from_list("seismic_red_both", new_colors_both)

y_ax = np.linspace(0.5,len(V_list)-0.5,len(V_list))
x_ax = np.linspace(0.5,len(N_list)-0.5,len(N_list))

plt.figure(dpi=500,figsize=(8*cm,8*cm))
plt.pcolormesh(np.transpose(MAP_tot_norm),edgecolors='w', linewidths=0.0,cmap = "bwr",vmin = -0.5,vmax=0.5)
plt.colorbar()
plt.gca().set_yticks(y_ax)
plt.gca().set_xticks(x_ax)
plt.ylabel('V')
plt.xlabel("N")
plt.gca().set_yticklabels(V_list)
plt.gca().set_xticklabels(N_list,rotation = 90)


plt.show()






#%% PLOT SINGLE CURVE

#%%
Vname_list = ['0003', '001', '003', '01', '030000000000000004']
Nname_list =[ '260', '510', '760', '1010', '1260', '1510', '1760', '2010']

i_V = 1
i_N = 5

print(Vname_list[i_V])
print(Nname_list[i_N])

#%%
dir_local = os.path.dirname(__file__)

name =    "Gaska2020_1111_MAP_run_autoV"+str(Vname_list[i_V])+"N"+str(Nname_list[i_N])
dir_input_file = '/home/cedrik/Documents/filaments-crosslinkers-projects/Gaska2020/'+name+'/'
name_input_file = name+'_s1'
extension_input_file = '.h5'

dir_output_file = dir_local
name_output_file = "output_test1"


bool_anim = 0

# print(name_save)




#%% loading data
tasks = d3.load_tasks_to_xarray(dir_input_file+name_input_file+extension_input_file) # Downloadig the files
# Vname_list = ['0025']

x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])

# x_tasks = np.array(tasks['n_Mab']['x'])
# t_tasks = np.array(tasks['n_Mab']['t'])/60



#%% Force calculation


Overlap = np.zeros(len(t_tasks))
N_Pab = np.zeros(len(t_tasks))
N_O = np.zeros(len(t_tasks))
N_s = np.zeros(len(t_tasks))


for i in range(len(t_tasks)):
    
    N_Pab[i] = integrate.simpson(tasks['Pab'][i],x_tasks)
    N_O[i] = integrate.simpson(tasks['Pab'][i]+(tasks['Pa'][i]+tasks['Pb'][i])*tasks['f_D'][i],x_tasks)
    N_s[i] = integrate.simpson((tasks['Pa'][i]+tasks['Pb'][i])*tasks['f_D'][i],x_tasks)

    Overlap[i] = integrate.simpson(tasks['f_D'][i],x_tasks)


#%% FIGURES

FB_tot = -4*1e-3*tasks['F_B'][~np.isnan(tasks['F_B'])][1:]
FB_ent = -4*1e-3*tasks['F_fB_ent'][~np.isnan(tasks['F_fB_ent'])][1:]
FB_fri = -4*1e-3*tasks['F_fB_fri'][~np.isnan(tasks['F_fB_fri'])][1:]




plt.figure(dpi=500,figsize=(8*cm,8*cm))
# plt.hlines(1, 0, 5,color="gray",linewidth = 2,linestyles="--",label="1")
plt.plot(Overlap[:len(FB_tot)],FB_tot/np.mean(FB_tot[0]),color="black",linewidth = 2,label = "Total")
plt.plot(Overlap[:len(FB_ent)],FB_ent/np.mean(FB_tot[0]),color="red",linewidth = 2,label = "Entropic")
plt.plot(Overlap[:len(FB_fri)],FB_fri/np.mean(FB_tot[0]),color="blue",linewidth = 2,label = "Friction")

# plt.ylim(0.5,1.5)
plt.xlim(-0.1,5.3)

plt.ylabel("Normalized Force")
plt.xlabel("Overlap Length ($\mu$m)")
plt.legend()
plt.savefig("plot_output_curves.png", dpi=500, bbox_inches="tight", transparent=True)

plt.show()



#%%
V_list = [ 0.003, 0.01, 0.03, 0.1, 0.3]

y_ax = np.linspace(0.5,len(V_list)-0.5,len(V_list))
x_ax = np.linspace(0.5,len(N_list)-0.5,len(N_list))

plt.figure(dpi=500,figsize=(8*cm,8*cm))
plt.pcolormesh(np.transpose(MAP_tot_norm),edgecolors='w', linewidths=0.0,cmap = "bwr",vmin = -0.5,vmax=0.5)
plt.colorbar()
plt.scatter(1+0.5,3+0.5,marker="x",color="black",s=50,linewidths=4)
plt.scatter(1+0.5,1+0.5,marker="x",color="black",s=50,linewidths=4)
plt.scatter(5+0.5,1+0.5,marker="x",color="black",s=50,linewidths=4)

plt.gca().set_yticks(y_ax)
plt.gca().set_xticks(x_ax)
plt.ylabel('V ($\mu$m.s$^1$)')
plt.xlabel("N")
plt.gca().set_yticklabels(V_list)
plt.gca().set_xticklabels(N_list,rotation = 90)
plt.savefig("plot_MAP_tot.png", dpi=500, bbox_inches="tight", transparent=True)


plt.show()




#%%
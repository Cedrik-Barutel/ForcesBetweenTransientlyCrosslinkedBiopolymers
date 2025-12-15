#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:42:24 2024

@author: cedrik
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate
import sys
import os
import logging

# local library
dir_local = os.path.dirname(__file__)
sys.path.append(dir_local)
lib_path = "/home/cedrik/Documents/filaments-crosslinkers-projects/lib"
sys.path.append(lib_path)

logger = logging.getLogger(__name__)
import lib_simulation as LS


#%%
# Read CSV file into pandas DataFrame
# name = 'Exp_1_OT.csv'
# name = 'Data_Kucera/Kucera_Data.xlsx - Fig. 2 & S2- Stretching in OT.csv'

name = 'Data_Kucera/pulling_exp/pulling_id_1.csv'


path = dir_local
df = pd.read_csv(path + '/'+name)

# Display the DataFrame (table)
print(df)

# Put the DataFram in something usable for numpy and me
data = np.transpose(np.array(df))



#%%
# Allocating arrays to specific data
T_raw = data[0] # Time
F_raw = data[1] # Force
L_raw = data[2] # Position



#%%
#Plotting the time to see if there is any problem
plt.figure(dpi=200)
plt.plot(T_raw,label="T_raw")
plt.plot(np.gradient(T_raw)*100,label="dt_raw")
plt.legend()
plt.show()

#%%
median_dt = np.median(np.gradient(T_raw))
DT = (T_raw[-1]-T_raw[0])

factor = 5
T_const = np.linspace(T_raw[0],T_raw[-1],factor*int(DT/median_dt))



#%% # convert the new time array to se same length as the old one
# T_const = LS.function_convertion_array(T_const, len(T_raw))

f_interpolate_L = interpolate.interp1d(T_raw, L_raw)
f_interpolate_F = interpolate.interp1d(T_raw, F_raw)

L_inter = f_interpolate_L(T_const)
F_inter = f_interpolate_F(T_const)



#%%


#%% Plotting the raw data

plt.figure(dpi = 300)
plt.plot(T_raw,L_raw,label = "raw",color="gray")
plt.plot(T_const,L_inter,label = "inter",color="black")
# plt.plot(T_const,L_inter,label = "raw",color="black")

plt.legend()
plt.xlabel("time")
plt.title("Raw data")
plt.show()

#%% Plotting the raw data

plt.figure(dpi = 300)
plt.plot(T_raw,F_raw,label = "raw",color="gray")
plt.plot(T_const,F_inter,label = "inter",color="black")

plt.legend()
plt.xlabel("time")
plt.title("Raw data")
plt.show()


#%% Plotting the raw data
#id_1 1 jump = start 7203 end 8700



i_start = 8700 # consecutiv jump
i_end = 13000

# i_start = 7203 #Single jump
# i_end = 8700
plt.figure(dpi = 300)
plt.plot(T_const,L_inter,label = "raw_corrected",color="gray")
plt.plot(T_const[i_start:i_end],L_inter[i_start:i_end],label = "interest",color="black")

plt.vlines(T_const[i_start], np.min((L_inter[i_start],L_inter[i_end])),np.max((L_inter[i_start],L_inter[i_end])),color="black",linestyle = "--")
plt.vlines(T_const[i_end],np.min((L_inter[i_start],L_inter[i_end])),np.max((L_inter[i_start],L_inter[i_end])),color="black",linestyle = "--")
plt.legend()
plt.xlabel("time")
plt.title("Raw data corrected")
plt.show()



#%% Plotting the raw data

plt.figure(dpi = 300)
plt.plot(T_const,F_inter,label = "raw_corrected",color="gray")
plt.plot(T_const[i_start:i_end],F_inter[i_start:i_end],label = "interest",color="black")

plt.vlines(T_const[i_start], np.min((F_inter[i_start],F_inter[i_end])),np.max((F_inter[i_start],F_inter[i_end])),color="black",linestyle = "--")
plt.vlines(T_const[i_end],np.min((F_inter[i_start],F_inter[i_end])),np.max((F_inter[i_start],F_inter[i_end])),color="black",linestyle = "--")
plt.legend()
plt.xlabel("time")
plt.title("Raw data corrected")
plt.show()



# %%

index_T_start_raw = np.abs(T_raw - T_const[i_start]).argmin()
index_T_end_raw = np.abs(T_raw - T_const[i_end]).argmin()


#%% Plotting the raw data ZOOM

plt.figure(dpi = 300)

plt.plot(T_const[i_start:i_end],F_inter[i_start:i_end],label = "interest",color="black")
plt.plot(T_raw[index_T_start_raw:index_T_end_raw],F_raw[index_T_start_raw:index_T_end_raw],label = "raw interest",color="black")

plt.legend()
plt.xlabel("time")
plt.xlim(T_const[i_start-10],T_const[i_end+10])
plt.title("Raw data corrected")
plt.show()

#%% Plotting the raw data

plt.figure(dpi = 300)
plt.plot(T_raw,L_raw,label = "raw",color="gray")
plt.plot(T_raw[index_T_start_raw:index_T_end_raw],L_raw[index_T_start_raw:index_T_end_raw],label = "interest",color="black")

plt.vlines(T_raw[index_T_start_raw], np.min((L_raw[index_T_start_raw],L_raw[index_T_end_raw])),np.max((L_raw[index_T_start_raw],L_raw[index_T_end_raw])),color="black",linestyle = "--")
plt.vlines(T_raw[index_T_end_raw],np.min((L_raw[index_T_start_raw],L_raw[index_T_end_raw])),np.max((L_raw[index_T_start_raw],L_raw[index_T_end_raw])),color="black",linestyle = "--")
plt.legend()
plt.xlabel("time")
plt.title("Raw data")
plt.show()

#%%
plt.figure(dpi = 300)
plt.plot(T_raw,F_raw,label = "raw",color="gray")
plt.plot(T_raw[index_T_start_raw:index_T_end_raw],F_raw[index_T_start_raw:index_T_end_raw],label = "interest",color="black")

plt.vlines(T_raw[index_T_start_raw], np.min((F_raw[index_T_start_raw],F_raw[index_T_end_raw])),np.max((F_raw[index_T_start_raw],F_raw[index_T_end_raw])),color="black",linestyle = "--")
plt.vlines(T_raw[index_T_end_raw],np.min((F_raw[index_T_start_raw],F_raw[index_T_end_raw])),np.max((F_raw[index_T_start_raw],F_raw[index_T_end_raw])),color="black",linestyle = "--")
plt.legend()
plt.xlabel("time")
plt.title("Raw data")
plt.show()


#%%
print("Delta T = "+str(T_const[i_end]-T_const[i_start]))
print("Delta L = "+str(L_inter[i_end]-L_inter[i_start]))
print("Delta T = "+str(T_raw[index_T_end_raw]-T_raw[index_T_start_raw]))
print("Delta L = "+str(L_raw[index_T_end_raw]-L_raw[index_T_start_raw]))


#%% Modifying the date to start from zero and adding zeros at the beggining and the end

Length_extension = int(0.01*len(L_inter[i_start:i_end]))
ext = np.ones(Length_extension)



F = np.concatenate((F_inter[i_start]*ext, F_inter[i_start:i_end],F_inter[i_end]*ext))
L = np.concatenate((L_inter[i_start]*ext, L_inter[i_start:i_end],L_inter[i_end]*ext))

if i_start>Length_extension:
    T = np.concatenate((T_const[i_start-Length_extension:i_start],T_const[i_start:i_end],np.max(T_const[i_start:i_end])+(T_const[-Length_extension:]-np.min(T_const[-Length_extension-1:]))) )
if i_start<=Length_extension:
    T = np.concatenate((T_const[i_start:i_start+Length_extension]-(T_const[i_start+Length_extension] - T_const[i_start]),T_const[i_start:i_end],np.max(T_const[i_start:i_end])+(T_const[-Length_extension:]-np.min(T_const[-Length_extension-1:]))) )

F = F
L = L-np.min(L)
T = T-np.min(T)



#%% same for raw data
Length_extension = int(0.01*len(L_raw[index_T_start_raw:index_T_end_raw]))
ext = np.ones(Length_extension)


F_raw = np.concatenate((F_raw[index_T_start_raw]*ext, F_raw[index_T_start_raw:index_T_end_raw],F_raw[index_T_end_raw]*ext))
L_raw = np.concatenate((L_raw[index_T_start_raw]*ext, L_raw[index_T_start_raw:index_T_end_raw],L_raw[index_T_end_raw]*ext))

if index_T_start_raw>Length_extension:
    T_raw = np.concatenate((T_raw[index_T_start_raw-Length_extension:index_T_start_raw],T_raw[index_T_start_raw:index_T_end_raw],np.max(T_raw[index_T_start_raw:index_T_end_raw])+(T_raw[-Length_extension:]-np.min(T_raw[-Length_extension-1:]))) )
if index_T_start_raw<=Length_extension:
    T_raw = np.concatenate((T_raw[index_T_start_raw:index_T_start_raw+Length_extension]-(T_raw[index_T_start_raw+Length_extension] - T_raw[index_T_start_raw]),T_raw[index_T_start_raw:index_T_end_raw],np.max(T_raw[index_T_start_raw:index_T_end_raw])+(T_raw[-Length_extension:]-np.min(T_raw[-Length_extension-1:]))) )

F_raw = F_raw
L_raw = L_raw-np.min(L_raw)
T_raw = T_raw-np.min(T_raw)


#%% Plotting the date we are going to use

plt.figure(dpi=200)
plt.plot(T,L,label = "Data of interest (corrected) ", color="black")
plt.plot(T_raw,L_raw,label = "Data of interest (raw)", color="black")

plt.legend()
plt.show()




#%%
sigma = 0.6
x=np.linspace(0,100,len(T))
G = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1*(1/2*sigma**2)*(x-50)**2)
plt.plot(G)
m_L = sp.signal.convolve(L-np.min(L),G,mode="same")/np.sum(G)




#%%
plt.figure(dpi=200)
plt.plot(T,L-np.min(L))
plt.plot(T,m_L)


plt.show()






#%%

V = np.gradient(m_L,T)

#%%

plt.figure(dpi = 300)
plt.plot(T,V,label = "Velocity(t)")

plt.legend()
plt.xlabel("time (ms)")
plt.show()


#%%

plt.figure(dpi = 300)
plt.plot(T[1:],sp.integrate.cumulative_simpson(V,x=T),label = "")
plt.plot(T,L,label = "")

plt.legend()
plt.xlabel("time (ms)")
plt.show()


#%%

plt.figure(dpi = 300)
plt.plot(T,V,label = "Velocity(t)")



plt.legend()
plt.xlabel("time (ms)")
# plt.xlim(2000,4000)
# plt.ylim(-0.001,0.001)

plt.show()



#%%
i_cut_left = 100
i_cut_right = -200



plt.figure(dpi = 300)
plt.plot(T,V,label = "Velocity(t)")
plt.vlines(T[i_cut_left],-1e-5,20e-5,color = "darkorange",linewidth=2)
plt.vlines(T[i_cut_right],-1e-5,20e-5,color = "darkorange",linewidth=2)

plt.legend()
plt.xlabel("time")
plt.title("velocity generated")
plt.ylim(1.1*np.min(V),1.1*np.max(V))
plt.show()




#%%

index_T_raw_cut_left = np.abs(T_raw - T[i_cut_left]).argmin()
index_T_raw_cut_right = np.abs(T_raw - T[i_cut_right]).argmin()



plt.figure(dpi = 300)
plt.plot(T,F,label = "Velocity(t)")
plt.vlines(T[i_cut_left],6,15,color = "blue",linewidth=2)
plt.vlines(T[i_cut_right],6,15,color = "blue",linewidth=2)


plt.plot(T_raw,F_raw)
plt.vlines(T_raw[index_T_raw_cut_left],6,14,color = "darkorange",linewidth=2)
plt.vlines(T_raw[index_T_raw_cut_right],6,14,color = "darkorange",linewidth=2)


plt.legend()
plt.xlabel("time")
plt.title("force_cut")
# plt.ylim(1.1*np.min(V),1.1*np.max(V))
plt.show()




#%%

V_cut = V[i_cut_left:i_cut_right]
F_cut = F[i_cut_left:i_cut_right]
T_cut = T[i_cut_left:i_cut_right]

F_raw_cut = F_raw[index_T_raw_cut_left:index_T_raw_cut_right]
T_raw_cut = T_raw[index_T_raw_cut_left:index_T_raw_cut_right]
L_raw_cut = L_raw[index_T_raw_cut_left:index_T_raw_cut_right]
#%%
plt.figure(dpi = 300)
plt.plot(T_cut,V_cut,label = "Velocity(t)")
plt.legend()
plt.xlabel("time")
plt.title("velocity generated")
plt.show()

#%%
V_save = V_cut
F_save = F_cut
T_save = T_cut-T_cut[0]

T_raw_save = T_raw_cut-T_raw_cut[0]
F_raw_save = F_raw_cut
L_raw_save = L_raw_cut-L_raw_cut[0]
#%%
dataset = pd.DataFrame({'Time': T_save , 'Velocity':V_save, 'Force':F_save}  )
dataset_raw = pd.DataFrame({'Time_raw':T_raw_save, 'Length_raw':L_raw_save, 'Force_raw':F_raw_save}  )
#%% SAVING THE FILES

name_save = "1112_VP_4_jump_s06"
format_ext = "csv"
dataset.to_csv(path + '/'+name_save +'.'+format_ext, index=False)


format_ext = "csv"
dataset_raw.to_csv(path + '/'+name_save+"_raw" +'.'+format_ext, index=False)

#%%






plt.figure(dpi = 200)
plt.plot(T_save,F_save,marker = "x",linewidth=0)
plt.plot(T_raw_save,F_raw_save,marker="x",linewidth=0)
plt.show()



#%%

plt.figure(dpi = 200)
# plt.plot(T_save,F_save,marker = "x",linewidth=0)
plt.plot(T_raw_save,F_raw_save,marker="x",linewidth=0)
plt.plot(T_raw_save,100*L_raw_save,marker="x",linewidth=0)
plt.plot(T_save,V_save*100000)
plt.show()





# SMALL_SIZE = 12
# MEDIUM_SIZE = 12
# BIGGER_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.figure(dpi = 500)
# plt.scatter(L,F,s=1,color="black")
# plt.xlabel("Position of the moving filament (um)")
# plt.ylabel("Force on the moving filament (pN)")

# plt.show()




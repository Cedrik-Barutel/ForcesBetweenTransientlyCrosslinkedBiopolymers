#%% IMPORT
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import datetime
from matplotlib.animation import FuncAnimation
from scipy import integrate
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import pandas as pd
import scipy as scipy
import logging
import os
import csv

import sys
# local library
dir_local = os.path.dirname(__file__)
sys.path.append(dir_local)
lib_path = "/home/cedrik/Documents/filaments-crosslinkers-projects/lib"
sys.path.append(lib_path)

logger = logging.getLogger(__name__)

import lib_simulation as LS

#%% VARIABLES
folder_name = "motor_on_filament_1112" # name of the simulation 
working_path = os.path.dirname(__file__)
folder = working_path  + "/"+ folder_name

N_save = 200 # number of save
timestep = 3e-2 #timestep
timestep0 = 5e-4
stop_time = 3*9*60 # max simulating time 
# Lx = 3 #length of the simulation box 4 jumps
Lx = 7 #length of the simulation box 1 jumps
Nx = 2**9

V=0.5
'Dimensions of the filaments'
# position of the filaments
xl_A = 1

xr_A = 6



Ly = LS.Coefficient("Ly",0.01) # Distance between the filaments
Lz = LS.Coefficient("Lz",0.01) # Witdh of the filaments

n_s = LS.Coefficient("n_s",125) # Lineic binsind site concentration

Eb_Pa = LS.Coefficient("Eb_Pa",5) # motors bound to (a)
Eb_Pb = LS.Coefficient("Eb_Pb",5) # motors bound to (b)
# Eb_Mab = LS.Coefficient("Eb_Mab",0) # motors bound to (ab)
Eb_Pab = LS.Coefficient("Eb_Pab",0) # passive bound to (ab) 

# Eb_Pa.v = -np.log(0.05/(1-0.05))
# Eb_Pb.v = -np.log(0.05/(1-0.05))
# Eb_Pab.v = -np.log(0.6/(1-0.6)*(1-0.6)/(1-0.6-0.05))


D_Ma = LS.Coefficient("D_Ma",0.001) # motors bound to (a)


# D_Pa = LS.Coefficient("D_Pa",0.0) # motors bound to (a)
# D_Pb = LS.Coefficient("D_Pb",0.0) # motors bound to (b)
# # D_Mab = LS.Coefficient("D_Mab",0) # motors bound to (ab)
# D_Pab = LS.Coefficient("D_Pab",0.0) # passive bound to (ab) 
# V_Ma = LS.Coefficient("V_Ma",0)
# V_Mb = LS.Coefficient("V_Mb",0)
V_Ma = LS.Coefficient("V_Ma",0.05)




Koff_1_Ma = LS.Coefficient("Koff_1_Ma",0.002777777777777777) # Rates of reaction 1 around Ma_eq
# Koff_2_Pb = LS.Coefficient("Koff_2_Pb",0.0) # Rates of reaction 2 around Mb_eq
# Koff_3_Pab = LS.Coefficient("Koff_3_Pab",0.0) # Rates of reaction 3 around Mab_eq
# Koff_4_Pab = LS.Coefficient("Koff_4_Pab",0.0) # Rates of reaction 4 around Mab_eq
# Koff_5_Pab = LS.Coefficient("Koff_5_Pab",0.0) # Rates of reaction 5 around Pab_eq

# Ma_ext = Ls.coefficient("Ma_ext",)


eta = LS.Coefficient("eta",0) # Shear viscosity
gamma = LS.Coefficient("gamma",0) # Viscous friction
#E = LS.Coefficient("E",20000) # Shear elasticity modulus
E = LS.Coefficient("E",0) # Shear elasticity modulus

act = LS.Coefficient("act",0) # Activity coefficient
#K = LS.Coefficient("K",1) # Convertion bwetween elastic strain and velocity
K = LS.Coefficient("K",0) # Convertion bwetween elastic strain and velocity
#G = LS.Coefficient("G",1) # Elastic relaxation rate (1/t)
G = LS.Coefficient("G",0) # Elastic relaxation rate (1/t)



#%%
h = 2e-3 # numerical help to avoid division by 0
h_log = 2*h # numerical help to log(0)

r = 50  # Stabilization with r*diffusion coefficient

# Phase field model 
li = 0.05 # length of the filaments interface
D_f = 0.2 # diffusion coefficient of the phase field
G_f = 1/18*li**2 # actual coefficient used


timestepper = d3.RK443 #time iteration scheme
dealias = 3/2 # anti aliasing factor


#%% DEDALUS BASIS, FUNCTION DEFINITION AND DEDALUS FIELD
# BUILDING DEDALUS COORDINATES AND BASIS
coords = d3.CartesianCoordinates('x')

dtype = np.float64
dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
x = dist.local_grids(xbasis)
ex = coords.unit_vector_fields(dist)

# DEFINING FUNCTIONS AND SUBTITUTION
dx = lambda A: d3.Differentiate(A, coords['x'])
ddx = lambda A: dx(dx(A))

# SETTING DEDALUS FIELDS

f_A = dist.Field(name='f_A',bases=(xbasis)) # filament A


Ma = dist.Field(name='Ma',bases=(xbasis)) # Concentration of 



# Velocities
V_A = dist.Field(name='V_A',bases=(xbasis)) # Velocity of A
V_B = dist.Field(name='V_B',bases=(xbasis)) # Velocity of B





#%% INITIAL CONDITIONS
'FILAMENTS PHASE FIELD'
f_A['g'] = LS.function_filament(f_A['g'], xl_A, xr_A, x[0], li)

'NUMBER OF PARTICLE'
# 1st guess on the number of particle definition

b = 0.0
a = 0.02
Ma['g'] = b*f_A['g']
# Ma['g'] = 0.8*np.exp(-30*x[0]**2)


#%%
plt.figure(dpi=200)
plt.plot(x[0],f_A['g'],label = "$f^{A}$")

plt.plot(x[0],Ma['g'],label = "$M^{a}$")


plt.ylim(-0.1,1.1)
plt.legend()
plt.show()



# %% EQUATIONS OF THE PROBLEM
# it's better to write the full equations at once instead of using variables

problem = d3.IVP([ f_A,
                    Ma,#, Mb, Mab, Pab,
                    # Pab,# Pa,Pb,
                    # Pa_eq_1, Pb_eq_2, Pab_eq_3, Pab_eq_4,
                    # C_Mab_Mab_inv,
                    # V_A,V_B,
                    # Overlap,
                    # F_A,F_B,
                    # Coeff_fri,
                    # grad_mu_fA,grad_mu_fB,
                    # F_fA_vis, F_fA_fri, F_fA_ent, F_fA_act, F_fA_ela,
                    # F_fB_vis, F_fB_fri, F_fB_ent, F_fB_act, F_fB_ela
                    ],
                  namespace=locals()) # Declaration of the problem variables

# - Cahn Hillard equation for the filaments - #
problem.add_equation("dt(f_A) +D_f*ddx(-2*f_A +G_f*ddx(f_A)) = D_f*ddx(4*(f_A)**3-6*(f_A)**2) -dx(f_A*V_A) ")


problem.add_equation("dt(Ma)"
                      "-r*D_Ma.v*ddx(Ma)" # stabilization
                      "-D_Ma.v*ddx(Ma)" # diffusion 
                      "+Koff_1_Ma.v*(Ma-0.24*f_A) " # chemical reactions
                      "="
                      "-V_Ma.v*dx(Ma*(f_A-Ma)/(h+f_A))"
                      "-r*D_Ma.v*ddx(Ma)" # stabilization
                      "-D_Ma.v*dx( (Ma/(h+f_A)  )*dx(f_A))" # 
                            )

# problem.add_equation("dt(Ma)"
#                       "-r*D_Ma.v*ddx(Ma)" # stabilization
#                       "-D_Ma.v*ddx(Ma)" # diffusion 
#                       # "+Koff_3_Pab.v*Pab +Koff_4_Pab.v*Pab" # chemical reactions
#                       "="
#                       "-V_Ma.v*dx(Ma*(1-Ma)/(h+1))"
#                       "-r*D_Ma.v*ddx(Ma)" # stabilization
#                       # "-D_Ma.v*dx( (Ma/(h+f_A)  )*dx(f_A))" # 
#                             )

# - Gradient of the chemical potential of the filaments - #
# problem.add_equation("grad_mu_fA = -1*( np.log((h_log+f_D)/(h_log+f_D-Pab))*dx(f_B) +f_B*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Pab)*dx(f_D-Pab)) + 1/(h+f_A-Pab)*dx(f_A-Pab) -1/(h+f_A-Pab-Pa)*dx(f_A-Pab-Pa)  )")
# problem.add_equation("grad_mu_fB = -1*( np.log((h_log+f_D)/(h_log+f_D-Pab))*dx(f_A) +f_A*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Pab)*dx(f_D-Pab)) + 1/(h+f_B-Pab)*dx(f_B-Pab) -1/(h+f_B-Pab-Pb)*dx(f_B-Pab-Pb)  )")

# - Coefficients - #
# problem.add_equation("Coeff_fri = Lz.v/Ly.v*eta.v*d3.Integrate((f_D )  ,('x'))")

# # - Integration of the forces - #
# problem.add_equation("F_fA_vis = -gamma.v*V_A  ")
# problem.add_equation("F_fA_fri = Coeff_fri*(V_B-V_A)")
# problem.add_equation("F_fA_ent = -n_s.v*d3.Integrate(f_A*( grad_mu_fA)  ,('x'))")
# problem.add_equation("F_fA_act = -act.v*d3.Integrate(f_A*(Mab)  ,('x'))")
# problem.add_equation("F_fA_ela = -E.v*Lz.v*d3.Integrate(f_A*(u_el)  ,('x'))")

# problem.add_equation("F_fB_vis = -gamma.v*V_B  ")
# problem.add_equation("F_fB_fri = -Coeff_fri*(V_B-V_A)")
# problem.add_equation("F_fB_ent = -n_s.v*d3.Integrate(f_B*( grad_mu_fB)  ,('x'))")
# problem.add_equation("F_fB_act =  act.v*d3.Integrate(f_B*(Mab)  ,('x'))")
# problem.add_equation("F_fB_ela =  E.v*Lz.v*d3.Integrate(f_B*(u_el)  ,('x'))")


# #In this experiment the A is fixed, and B is force-free.
# # - Forces - #
# problem.add_equation("F_A = F_fA_vis +F_fA_fri +F_fA_ent +F_fA_act +F_fA_ela") 
# problem.add_equation("F_B = F_fB_vis +F_fB_fri +F_fB_ent +F_fB_act +F_fB_ela") 

# - Velocities - #
# problem.add_equation("V_B = v_B")
# problem.add_equation("V_A = v_A")

# problem.add_equation("dt(Overlap)=V_A-V_B")




#%%


#%% BUILDING SOLVER
solver = problem.build_solver(timestepper,ncc_cutoff=1e-4)
solver.stop_sim_time = stop_time


#%%
'Setting the paramters used and setting the save of the simulation'
date = datetime.datetime.now()
name = str(folder)

analysis = solver.evaluator.add_file_handler(folder, sim_dt=stop_time/N_save)
analysis.add_tasks(solver.state, layout='g') # Save all variables of the problem
# analysis.add_task(n_A,layout = 'g',name = 'n_A') #Save a specific variable


ListCoeffSave = {obj for name, obj in globals().items() if isinstance(obj, LS.Coefficient)}
with open( folder+ "/" +"sparameters_"+folder_name+'.csv', 'w', newline='') as filecsv:
    fieldnames = ['name','value']
    writer = csv.DictWriter(filecsv, fieldnames=fieldnames) 
    writer.writeheader()
    for Coeff in ListCoeffSave:   
        LS.function_save_parameters(writer, fieldnames, Coeff)


# %% Starting the main loop
print("Start")
j=0
t=0
z=0
T_N0 = datetime.datetime.now()
while solver.proceed:

    t=t+1   
    
    hk = 1e-3
    # timestep = 1e-4*(-np.log((0.1+abs(vA(solver.sim_time))/0.0864)/(1-abs(vA(solver.sim_time))/0.0864))*0.1+1)
    solver.step(timestep) # solving the equations   
    z=z+timestep
    # print(z)
    if z/(stop_time/(N_save)) >= 1 :
        z=0
        j=j+1
        T_N1 = datetime.datetime.now()
        T_LEFT = (T_N1-T_N0)*(N_save-j)
        logger.info('%i/%i, T=%0.2e, t_left = %s , dt=%0.2e' %(j,N_save,solver.sim_time,str(T_LEFT),timestep))
        T_N0 = datetime.datetime.now()
        
    
        if j%1  == 0 and j<10 or j%20 == 0:
                f_A.change_scales(1)
                Ma.change_scales(1)


                #Mb.change_scales(1)
                plt.figure(dpi=100)
                # plt.title(V_B['g'])
                plt.plot(x[0],f_A['g'],color = 'blue',alpha = 1)
                plt.plot(x[0],Ma['g'],color = 'red',alpha = 1)



                # plt.plot(x[0],grad_mu_fB['g']/np.max(np.abs(grad_mu_fB['g'])),color = 'black',label = "$f$")
                # plt.ylim(-0.1,0.2)

                
                plt.legend()
                plt.show()

#%%

# %% Getting the saved files
# tasks = d3.load_tasks_to_xarray(folder +"/"+folder_name+"_s1.h5") # Downloadig the files
# x_tasks = np.array(tasks['n_D']['x'])
# t_tasks = np.array(tasks['n_D']['t'])
print(folder +"/"+folder_name+"_s1.h5")
print("\nduration:")
print( T_N1-date)


#%%
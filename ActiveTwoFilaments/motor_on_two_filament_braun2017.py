#%% IMPORT
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import datetime
from matplotlib.animation import FuncAnimation
from scipy import integrate
from scipy.interpolate import UnivariateSpline
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
folder_name = "Braun2017_1112" # name of the simulation 
working_path = os.path.dirname(__file__)
folder = working_path  + "/"+ folder_name

N_save = 500 # number of save
timestep = 1e-2 #timestep
stop_time = 40*60 # max simulating time 
Lx = 10 #length of the simulation box
Nx = 2**9


'Dimensions of the filaments'
# position of the filaments
xl_A = -3
# xr_A = 0.77
# xl_B = -0.77
xr_A = 1
xl_B = -1
# xl_B = -5
# xl_B = -6

xr_B = 2

LA = xr_A-xl_A
LB = xr_B-xl_B


Ly = LS.Coefficient("Ly",0.02) # Distance between the filaments
Lz = LS.Coefficient("Lz",0.02) # Witdh of the filaments

n_s = LS.Coefficient("n_s",200) # Lineic binsind site concentration

Eb_Ma = LS.Coefficient("Eb_Ma",0) # motors bound to (a)
Eb_Mb = LS.Coefficient("Eb_Mb",0) # motors bound to (b)
Eb_Mab = LS.Coefficient("Eb_Mab",0) # motors bound to (ab)
Eb_Pa = LS.Coefficient("Eb_Pa",0) # motors bound to (a)
Eb_Pb = LS.Coefficient("Eb_Pb",0) # motors bound to (b)
Eb_Pab = LS.Coefficient("Eb_Pab",-1) # passive bound to (ab) 

D_Ma = LS.Coefficient("D_Ma",0.05) # motors bound to (a)
D_Mb = LS.Coefficient("D_Mb",0.05) # motors bound to (b)
D_Mab = LS.Coefficient("D_Mab",0.001/2) # motors bound to (ab)

D_Pa = LS.Coefficient("D_Pa",0.09) # motors bound to (a)
D_Pb = LS.Coefficient("D_Pb",0.09) # motors bound to (b)
D_Pab = LS.Coefficient("D_Pab",0.011) # passive bound to (ab) 

V_Ma = LS.Coefficient("V_Ma",0)
V_Mb = LS.Coefficient("V_Mb",0)
V_Mab = LS.Coefficient("V_Mab",0)



Koff_1_Ma = LS.Coefficient("Koff_1_Ma",0.1) # Rates of reaction 1 around Ma_eq
Koff_2_Mb = LS.Coefficient("Koff_2_Mb",0.1) # Rates of reaction 2 around Mb_eq
Koff_3_Mab = LS.Coefficient("Koff_3_Mab",30) # Rates of reaction 3 around Mab_eq
Koff_4_Mab = LS.Coefficient("Koff_4_Mab",30) # Rates of reaction 4 around Mab_eq



# Koff_5_Pa = LS.Coefficient("Koff_5_Pa",1/500) # Rates of reaction 1 around Ma_eq
# Koff_6_Pb = LS.Coefficient("Koff_6_Pb",1/500) # Rates of reaction 2 around Mb_eq
# Koff_7_Pab = LS.Coefficient("Koff_7_Pab",0.5) # Rates of reaction 3 around Mab_eq MEASURED
# Koff_8_Pab = LS.Coefficient("Koff_8_Pab",0.5) # Rates of reaction 4 around Mab_eq MEASURED
# Koff_1_Mab = LS.Coefficient("Koff_1_Mab",0.002777777777777777) # Rates of reaction 4 around Mab_eq MEASURED


eta = LS.Coefficient("eta",50) # Shear viscosity
gamma = LS.Coefficient("gamma",100) # Viscous friction
E = LS.Coefficient("E",0) # Shear elasticity modulus
# act = LS.Coefficient("act",288) # Activity coefficient constant
act = LS.Coefficient("act",1.5) # Activity coefficient withmotors
# act = LS.Coefficient("act",2) # Activity coefficient withmotors

K = LS.Coefficient("K",0) # Convertion bwetween elastic strain and velocity
G = LS.Coefficient("G",0) # Elastic relaxation





#%%
h = 2e-3 # numerical help to avoid division by 0
h_log = 2*h # numerical help to log(0)

r = 50  # Stabilization with r*diffusion coefficient

# Phase field model 
li = 0.2 # length of the filaments interface
D_f = 0.2 # diffusion coefficient of the phase field
G_f = 1/18*li**2 # actual coefficient used


timestepper = d3.RK443 #time iteration scheme
dealias = 3/2 # anti aliasing factor




#%% DEDALUS BASIS, FUNCTION DEFINITION AND DEDALUS FIELD
# BUILDING DEDALUS COORDINATES AND BASIS
coords = d3.CartesianCoordinates('x')

dtype = np.float64
dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
x = dist.local_grids(xbasis)
ex = coords.unit_vector_fields(dist)

# DEFINING FUNCTIONS AND SUBTITUTION
dx = lambda A: d3.Differentiate(A, coords['x'])
ddx = lambda A: dx(dx(A))

# SETTING DEDALUS FIELDS

f_A = dist.Field(name='f_A',bases=(xbasis)) # filament A
f_B = dist.Field(name='f_B',bases=(xbasis)) # filament B
f_D = dist.Field(name='f_D',bases=(xbasis)) # Overlap 


Ma = dist.Field(name='Ma',bases=(xbasis)) # Concentration of 
Mb = dist.Field(name='Mb',bases=(xbasis)) # Concentration of 
Mab = dist.Field(name='Mab',bases=(xbasis)) # Concentration of 

Pa = dist.Field(name='Pa',bases=(xbasis)) # Concentration of 
Pb = dist.Field(name='Pb',bases=(xbasis)) # Concentration of 
Pab = dist.Field(name='Pab',bases=(xbasis)) # Concentration of 


# Equilibirum densities
Ma_eq_1 = dist.Field(name='Ma_eq_1',bases=(xbasis)) # Equilibrium Concentration of Ma 
Mb_eq_2 = dist.Field(name='Mb_eq_2',bases=(xbasis)) # Concentration of 
Mab_eq_3 = dist.Field(name='Mab_eq_3',bases=(xbasis)) # Concentration of
Mab_eq_4 = dist.Field(name='Mab_eq_4',bases=(xbasis)) # Concentration of

Pa_eq_5 = dist.Field(name='Pa_eq_5',bases=(xbasis)) # Equilibrium Concentration of Ma 
Pb_eq_6 = dist.Field(name='Pb_eq_6',bases=(xbasis)) # Concentration of 
Pab_eq_7 = dist.Field(name='Pab_eq_7',bases=(xbasis)) # Concentration of
Pab_eq_8 = dist.Field(name='Pab_eq_8',bases=(xbasis)) # Concentration of  
 

# 
C_Mab_Mab_inv = dist.Field(name='C_Mab_Mab_inv',bases=(xbasis))
C_Mab_Pab = dist.Field(name='C_Mab_Pab',bases=(xbasis))
C_Mab_fA = dist.Field(name='C_Mab_fA',bases=(xbasis))
C_Mab_fB = dist.Field(name='C_Mab_fB',bases=(xbasis))

C_Pab_Pab = dist.Field(name='C_Pab_Pab',bases=(xbasis))
C_Pab_Mab = dist.Field(name='C_Pab_Mab',bases=(xbasis))
C_Pab_fA = dist.Field(name='C_Pab_fA',bases=(xbasis))
C_Pab_fB = dist.Field(name='C_Pab_fB',bases=(xbasis))

# Coefficient for the forces
Coeff_fri = dist.Field(name='Coeff_fri')

# Velocities
V_A = dist.Field(name='V_A') # Velocity of A
V_B = dist.Field(name='V_B') # Velocity of B

grad_mu_fA = dist.Field(name='grad_mu_fA',bases=(xbasis))
grad_mu_fB = dist.Field(name='grad_mu_fB',bases=(xbasis))

u_el = dist.Field(name='u_el',bases=(xbasis))


F_fA_vis = dist.Field(name='F_fA_vis') # viscous friction force
F_fA_fri = dist.Field(name='F_fA_fri') # shear friction force
F_fA_ent = dist.Field(name='F_fA_ent') # entropic force
F_fA_act = dist.Field(name='F_fA_act') # active force
F_fA_ela = dist.Field(name='F_fA_ela') # elastic force

F_fB_vis = dist.Field(name='F_fB_vis') # viscous friction force
F_fB_fri = dist.Field(name='F_fB_fri') # shear friction force
F_fB_ent = dist.Field(name='F_fB_ent') # entropic force
F_fB_act = dist.Field(name='F_fB_act') # active force
F_fB_ela = dist.Field(name='F_fB_ela') # elastic force


Coeff_act = dist.Field(name='Coeff_act')

F_A = dist.Field(name='F_A')
F_B = dist.Field(name='F_B')


#%% INITIAL CONDITIONS
'FILAMENTS PHASE FIELD'
f_A['g'] = LS.function_filament(f_A['g'], xl_A, xr_A, x[0], li)
f_B['g'] = LS.function_filament(f_B['g'], xl_B, xr_B, x[0], li)
f_D['g'] = np.minimum(f_A['g'],f_B['g'])

'NUMBER OF PARTICLE'
# 1st guess on the number of particle definition



Mab['g'] = 0.1*f_D['g']





#%%
# Pab['g']=0.3*f_D['g']-0.05*(x[0]-(xl_B+xr_B)/2)*f_D['g']
# Mab['g']=0.0*f_D['g']
# Pa['g']=0.0*f_D['g']
# Pb['g']=0.0*f_D['g']
# Pab['g']=0.0*f_D['g']



#%%
print(integrate.simpson(Mab['g'],x[0]))
#%%
plt.figure(dpi=200)
plt.title("Motors")
plt.plot(x[0],f_A['g'],label = "$f^{A}$")
plt.plot(x[0],f_B['g'],label = "$f^{B}$")
plt.plot(x[0],Ma['g'],label = "$M^{a}$")
plt.plot(x[0],Mb['g'],label = "$M^{b}$")
plt.plot(x[0],Mab['g'],label = "$M^{ab}$")
# plt.plot(x[0],(f_A['g']-Mab['g'])/(1+np.exp(Eb_Ma.v)),label = "$M^{ab}$")
# plt.plot(x[0],(f_A['g']-Pab['g']-Mab['g'])/(1+np.exp(Eb_Ma.v)),label = "$M^{ab}$")

plt.plot(x[0],Ma_eq_1['g'],label = "$M^{a}$",alpha = 0.3)
plt.plot(x[0],Mb_eq_2['g'],label = "$M^{b}$",alpha = 0.3)
plt.plot(x[0],Mab_eq_3['g'],label = "$M^{ab}_{eq}$",alpha = 0.3)
plt.plot(x[0],Mab_eq_4['g'],label = "$M^{ab}_{eq}$",alpha = 0.3)

plt.hlines(0.6, -20,0)
# plt.hlines(0.1, -20,0)

plt.ylim(-0.1,1.1)
plt.legend()
plt.show()






# %% EQUATIONS OF THE PROBLEM
# it's better to write the full equations at once instead of using variables

problem = d3.IVP([ f_A,f_B,f_D,
                    Mab,
                    # Ma_eq_1, Mb_eq_2, Mab_eq_3, Mab_eq_4,
                    # Pa_eq_5, Pb_eq_6, Pab_eq_7, Pab_eq_8,
                    # C_Mab_Mab_inv,
                    V_A,V_B,
                    F_A,F_B,
                    Coeff_fri,
                    grad_mu_fA,grad_mu_fB,
                    # u_el,
                    F_fA_vis, F_fA_fri, F_fA_ent, F_fA_act,
                    F_fB_vis, F_fB_fri, F_fB_ent, F_fB_act],
                  namespace=locals()) # Declaration of the problem variables

# - Cahn Hillard equation for the filaments - #
problem.add_equation("dt(f_A) +D_f*ddx(-2*f_A +G_f*ddx(f_A)) = D_f*ddx(4*(f_A)**3-6*(f_A)**2) -dx(f_A*V_A) ")
problem.add_equation("dt(f_B) +D_f*ddx(-2*f_B +G_f*ddx(f_B)) = D_f*ddx(4*(f_B)**3-6*(f_B)**2) -dx(f_B*V_B)")
problem.add_equation("f_D = f_A*f_B")

# - Equation of the particles - #

problem.add_equation("dt(Mab)"
                      "-r*D_Mab.v*ddx(Mab)" # stabilization
                      "-D_Mab.v*ddx(Mab)" # diffusion 
                      # "+Koff_1_Mab.v*Mab" # chemical reactions
                      "="
                      "-r*D_Mab.v*ddx(Mab)" # stabilization
                      "-D_Mab.v*dx( (f_B*Mab/(h+f_D)  )*dx(f_A))" # 
                      "-D_Mab.v*dx( (f_A*Mab/(h+f_D)  )*dx(f_B))" # 
                      "-dx(Mab*0.5*(V_A+V_B))" # convected flux
                      "+Koff_3_Mab.v*Mab_eq_3  +Koff_4_Mab.v*Mab_eq_4"
                      ) # chemical reactions





# - Equation of the mobilities - #


# # - Equation of the equilibrium concentrations - #
# problem.add_equation("Ma_eq_1 = (f_A-Mab-Pab-Pa)/(1+np.exp(Eb_Ma.v))")
# problem.add_equation("Mb_eq_2 = (f_B-Mab-Pab-Pb)/(1+np.exp(Eb_Mb.v))")

# problem.add_equation("Mab_eq_3 = 0.5*(1-Pab+Ma*np.exp(-(Eb_Mab.v-Eb_Ma.v))-np.sqrt( (1-Pab+Ma*np.exp(-(Eb_Mab.v-Eb_Ma.v)))**2 -4*Ma*np.exp(-(Eb_Mab.v-Eb_Ma.v))*(f_B-Pab-Mb-Pb)))")
# problem.add_equation("Mab_eq_4 = 0.5*(1-Pab+Mb*np.exp(-(Eb_Mab.v-Eb_Mb.v))-np.sqrt( (1-Pab+Mb*np.exp(-(Eb_Mab.v-Eb_Mb.v)))**2 -4*Mb*np.exp(-(Eb_Mab.v-Eb_Mb.v))*(f_A-Pab-Ma-Pa)))")

# problem.add_equation("Pa_eq_5 = (f_A-Mab-Pab-Ma)/(1+np.exp(Eb_Pa.v))")
# problem.add_equation("Pb_eq_6 = (f_B-Mab-Pab-Mb)/(1+np.exp(Eb_Pb.v))")

# problem.add_equation("Pab_eq_7 = 0.5*(1-Mab+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1-Mab+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B-Mab-Mb-Pb)))")
# problem.add_equation("Pab_eq_8 = 0.5*(1-Mab+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1-Mab+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A-Mab-Ma-Pa)))")


# - Equation of elastic stress - #
# problem.add_equation("dt(u_el) + G.v*u_el = K.v*(V_B-V_A)/Ly.v  ")


# - Gradient of the chemical potential of the filaments - #
problem.add_equation("grad_mu_fA = -1*( np.log((h_log+f_D)/(h_log+f_D-Mab))*dx(f_B) +f_B*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Mab)*dx(f_D-Mab))  )")
problem.add_equation("grad_mu_fB = -1*( np.log((h_log+f_D)/(h_log+f_D-Mab))*dx(f_A) +f_A*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Mab)*dx(f_D-Mab))  )")

# - Coefficients - #
problem.add_equation("Coeff_fri = Lz.v/Ly.v*eta.v*n_s.v*d3.Integrate((Mab )  ,('x'))")

# - Integration of the forces - #
problem.add_equation("F_fA_vis = -gamma.v*V_A  ")
problem.add_equation("F_fA_fri = Coeff_fri*(V_B-V_A)")
problem.add_equation("F_fA_ent = -n_s.v*d3.Integrate(f_A*( grad_mu_fA)  ,('x'))")
problem.add_equation("F_fA_act = -Coeff_act*n_s.v*d3.Integrate((Mab)  ,('x'))")

problem.add_equation("F_fB_vis = -gamma.v*V_B  ")
problem.add_equation("F_fB_fri = -Coeff_fri*(V_B-V_A)")
problem.add_equation("F_fB_ent = -n_s.v*d3.Integrate(f_B*( grad_mu_fB)  ,('x'))")
problem.add_equation("F_fB_act =  Coeff_act*n_s.v*d3.Integrate((Mab)  ,('x'))")




# - Forces - #
problem.add_equation("F_A = 0") 
problem.add_equation("F_B = 0") 

# - Velocities - #
problem.add_equation("V_B = 1/(gamma.v+Coeff_fri)*( F_fB_ent +F_fB_act)")
problem.add_equation("V_A = 0*1/(gamma.v+Coeff_fri)*( F_fA_ent +F_fA_act)")


#%%
u_el['g'] = 0



Coeff_fri['g'] = Lz.v/Ly.v*eta.v*(d3.Integrate((Mab ),('x'))).evaluate()['g']


#%%


#%% BUILDING SOLVER
solver = problem.build_solver(timestepper,ncc_cutoff=1e-4)
solver.stop_sim_time = stop_time


#%%
'Setting the paramters used and setting the save of the simulation'
date = datetime.datetime.now()
name = str(folder)

analysis = solver.evaluator.add_file_handler(folder, sim_dt=stop_time/N_save, max_writes=N_save)
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
T_N0 = datetime.datetime.now()
while solver.proceed:
    t=t+1   
    # V_A['g'] = 0 
    # V_B['g'] = 0 
    Coeff_act['g'] = act.v*(0.5+0.5*np.tanh(0.05*(solver.sim_time-13*60)))*(0.5+0.5*np.tanh(-0.05*(solver.sim_time-35*60)))           
    solver.step(timestep) # solving the equations   
    
    if solver.iteration % int(stop_time/(N_save*timestep)) == 0 :
        j=j+1
        T_N1 = datetime.datetime.now()
        T_LEFT = (T_N1-T_N0)*(N_save-j)
        logger.info('%i/%i, T=%0.2e, t_left = %s' %(j,N_save,solver.sim_time,str(T_LEFT)))
        T_N0 = datetime.datetime.now()
        #print(V_B['g'][20])
        if j%1  == 0 and j<10 or j%10 == 0:
                f_A.change_scales(1)
                f_B.change_scales(1)
                # f_D.change_scales(1)

                #Mab.change_scales(1)
                #Pab.change_scales(1)
                # Ma.change_scales(1)
                # Ma_eq_1.change_scales(1)
                
                # Mb.change_scales(1)
                # Mb_eq_2.change_scales(1)
                
                
                
                Mab.change_scales(1)
                # Mab_eq_3.change_scales(1)
                # Mab_eq_4.change_scales(1)

                # Mab_eq_2.change_scales(1)
                
                # grad_mu_fB.change_scales(1)
                # C_Mab_fA.change_scales(1)

                # C_Mab_Mab_inv.change_scales(1)
                # Pa.change_scales(1)
                # Pb.change_scales(1)

                # Pab.change_scales(1)


                # fig, axs = plt.subplots(2)
                # fig.suptitle(V_B['g'][0])
                # axs[0].plot(x[0],f_A['g'],color = 'blue',alpha = 0.5)
                # axs[0].plot(x[0],f_B['g'],color = 'red',alpha = 0.5)
                # axs[0].plot(x[0],Ma['g'],color = 'blue',label = "$M^{a}$")
                # axs[0].plot(x[0],Mb['g'],color = 'red',label = "$M^{b}$")
                # axs[0].plot(x[0],Mab['g'],color = 'purple',label = "$M^{ab}$")
                
                # axs[1].plot(x[0],f_A['g'],color = 'blue',alpha = 0.5)
                # axs[1].plot(x[0],f_B['g'],color = 'red',alpha = 0.5)
                # axs[1].plot(x[0],Pa['g'],color = 'blue',label = "$P^{a}$")
                # axs[1].plot(x[0],Pb['g'],color = 'red',label = "$P^{b}$")
                # axs[1].plot(x[0],Pab['g'],color = 'violet',label = "$P^{ab}$")

                #Mb.change_scales(1)
                plt.title(V_B['g'])
                plt.plot(x[0],f_A['g'],color = 'blue',alpha = 0.5)
                plt.plot(x[0],f_B['g'],color = 'red',alpha = 0.5)
                # plt.plot(x[0],Ma['g'],color = 'blue',label = "$M^{a}$")
                # plt.plot(x[0],Mb['g'],color = 'red',label = "$M^{b}$")
                plt.plot(x[0],Mab['g'],color = 'purple',label = "$M^{ab}$")

                # plt.plot(x[0],Ma_eq_1['g'],color = 'blue',label = "$M^{a}_{eq}(1)$",linestyle="--")
                # plt.plot(x[0],Mb_eq_2['g'],color = 'red',label = "$M^{b}_{eq}(2)$",linestyle="--")
                # plt.plot(x[0],Mab_eq_3['g'],color = 'purple',label = "$M^{ab}_{eq}(3)$",linestyle="--")
                # plt.plot(x[0],Mab_eq_4['g'],color = 'purple',label = "$M^{ab}_{eq}(4)$",linestyle="--")
                
                # plt.plot(x[0],Pa['g'],color = 'blue',label = "$P^{a}$")
                # plt.plot(x[0],Pb['g'],color = 'red',label = "$P^{b}$")

                # plt.plot(x[0],Pab['g'],color = 'violet',label = "$P^{ab}$")
                
                # plt.plot(x[0],grad_mu_fB['g']/np.max(np.abs(grad_mu_fB['g'])),color = 'black',label = "$f$")
                plt.ylim(-0.1,1.1)
                
                plt.show()

#%%

# %% Getting the saved files
tasks = d3.load_tasks_to_xarray(folder +"/"+folder_name+"_s1.h5") # Downloadig the files
x_tasks = np.array(tasks['f_A']['x'])
t_tasks = np.array(tasks['f_A']['t'])
print(folder +"/"+folder_name+"_s1.h5")
print("\nduration:")
print( T_N1-date)


#%%
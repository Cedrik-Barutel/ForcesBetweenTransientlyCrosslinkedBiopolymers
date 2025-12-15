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
folder_name = "Kucera2021_1212_4_fine_s1_k1000_N2000" # name of the simulation 
working_path = os.path.dirname(__file__)
folder = working_path  + "/"+ folder_name

N_save = 2000 # number of save
timestep = 5e-4 #timestep
timestep0 = 2e-4
stop_time = 5 # max simulating time 
Lx = 3 #length of the simulation box 4 jumps
# Lx = 1.5 #length of the simulation box 1 jumps1*
Nx = 2**9

V=0.5
'Dimensions of the filaments'
# position of the filaments
xl_A = -1.1 #jumps
# xl_A = -0.5 #1 jump

# xr_A = 0.21/2*0.95
# xl_B = -0.21/2*0.95

# xr_A = 0.18517008923745493*0.7
# xl_B = -0.18517008923745493*0.7

xr_A = 0.275*0.85
xl_B = -0.275*0.85


xr_B = 1.1 #4jumps
# xr_B = 0.5 #1 jump

LA = xr_A-xl_A
LB = xr_B-xl_B


Ly = LS.Coefficient("Ly",0.01) # Distance between the filaments
Lz = LS.Coefficient("Lz",0.01) # Witdh of the filaments

n_s = LS.Coefficient("n_s",400*5*2) # Lineic binsind site concentration

Eb_Pa = LS.Coefficient("Eb_Pa",5) # motors bound to (a)
Eb_Pb = LS.Coefficient("Eb_Pb",5) # motors bound to (b)
# Eb_Mab = LS.Coefficient("Eb_Mab",0) # motors bound to (ab)
Eb_Pab = LS.Coefficient("Eb_Pab",0) # passive bound to (ab) 

# Eb_Pa.v = -np.log(0.05/(1-0.05))double
# Eb_Pb.v = -np.log(0.05/(1-0.05))
# Eb_Pab.v = -np.log(0.6/(1-0.6)*(1-0.6)/(1-0.6-0.05))


D_Pa = LS.Coefficient("D_Pa",0.0088) # motors bound to (a)
D_Pb = LS.Coefficient("D_Pb",0.0088) # motors bound to (b)
# D_Mab = LS.Coefficient("D_Mab",0) # motors bound to (ab)
D_Pab = LS.Coefficient("D_Pab",0.0088/2) # passive bound to (ab) 
# D_Pab = LS.Coefficient("D_Pab",0.1*0.0088) # passive bound to (ab) 


# D_Pa = LS.Coefficient("D_Pa",0.0) # motors bound to (a)
# D_Pb = LS.Coefficient("D_Pb",0.0) # motors bound to (b)
# # D_Mab = LS.Coefficient("D_Mab",0) # motors bound to (ab)
# D_Pab = LS.Coefficient("D_Pab",0.0) # passive bound to (ab) 
# V_Ma = LS.Coefficient("V_Ma",0)
# V_Mb = LS.Coefficient("V_Mb",0)
# V_Mab = LS.Coefficient("V_Mab",0)




Koff_1_Pa = LS.Coefficient("Koff_1_Pa",0.06) # Rates of reaction 1 around Ma_eq
Koff_2_Pb = LS.Coefficient("Koff_2_Pb",0.06) # Rates of reaction 2 around Mb_eq
Koff_3_Pab = LS.Coefficient("Koff_3_Pab",0.06*1000) # Rates of reaction 3 around Mab_eq
Koff_4_Pab = LS.Coefficient("Koff_4_Pab",0.06*1000) # Rates of reaction 4 around Mab_eq
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
h = 5e-3 # numerical help to avoid division by 0
h_log = 2*h # numerical help to log(0)

r = 500  # Stabilization with r*diffusion coefficient

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


Pa = dist.Field(name='Pa',bases=(xbasis)) # Concentration of 
Pb = dist.Field(name='Pb',bases=(xbasis)) # Concentration of 
Mab = dist.Field(name='Mab',bases=(xbasis)) # Concentration of 
Pab = dist.Field(name='Pab',bases=(xbasis)) # Concentration of 


# Equilibirum densities
Pa_eq_1 = dist.Field(name='Ma_eq_1',bases=(xbasis)) # Equilibrium Concentration of Ma 
Pb_eq_2 = dist.Field(name='Mb_eq_2',bases=(xbasis)) # Concentration of 
Pab_eq_3 = dist.Field(name='Mab_eq_3',bases=(xbasis)) # Concentration of
Pab_eq_4 = dist.Field(name='Mab_eq_4',bases=(xbasis)) # Concentration of 
 
Pab_eq_5 = dist.Field(name='Pab_eq_5',bases=(xbasis)) # Concentration of 

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
V_A = dist.Field(name='V_A',bases=(xbasis)) # Velocity of A
V_B = dist.Field(name='V_B',bases=(xbasis)) # Velocity of B

Overlap = dist.Field(name='Overlap',bases=(xbasis)) # Position of A


grad_mu_fA = dist.Field(name='grad_mu_fA',bases=(xbasis))
grad_mu_fB = dist.Field(name='grad_mu_fB',bases=(xbasis))

u_el = dist.Field(name='u_el',bases=(xbasis))


F_fA_vis = dist.Field(name='F_fA_vis',bases=(xbasis)) # viscous friction force
F_fA_fri = dist.Field(name='F_fA_fri',bases=(xbasis)) # shear friction force
F_fA_ent = dist.Field(name='F_fA_ent',bases=(xbasis)) # entropic force
F_fA_act = dist.Field(name='F_fA_act',bases=(xbasis)) # active force
F_fA_ela = dist.Field(name='F_fA_ela',bases=(xbasis)) # elastic force

F_fB_vis = dist.Field(name='F_fB_vis',bases=(xbasis)) # viscous friction force
F_fB_fri = dist.Field(name='F_fB_fri',bases=(xbasis)) # shear friction force
F_fB_ent = dist.Field(name='F_fB_ent',bases=(xbasis)) # entropic force
F_fB_act = dist.Field(name='F_fB_act',bases=(xbasis)) # active force
F_fB_ela = dist.Field(name='F_fB_ela',bases=(xbasis)) # elastic force


F_A = dist.Field(name='F_A',bases=(xbasis))
F_B = dist.Field(name='F_B',bases=(xbasis))

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

F_A = dist.Field(name='F_A')
F_B = dist.Field(name='F_B')

V_A = dist.Field(name='V_A') # Velocity of A
V_B = dist.Field(name='V_B') # Velocity of B



#%% INITIAL CONDITIONS
'FILAMENTS PHASE FIELD'
f_A['g'] = LS.function_filament(f_A['g'], xl_A, xr_A, x[0], li)
f_B['g'] = LS.function_filament(f_B['g'], xl_B, xr_B, x[0], li)
f_D['g'] = np.minimum(f_A['g'],f_B['g'])

'NUMBER OF PARTICLE'
# 1st guess on the number of particle definition

b = 0.533
a = 0.02

# b = 0.6093131458372931*0.95
# a = 0.2
Pab['g'] = b*f_D['g']
Pa['g'] = a*(f_A['g']-Pab['g'])
Pb['g'] = a*(f_B['g']-Pab['g'])

Eb_Pa.v = -np.log(a/(1-a))
Eb_Pb.v = -np.log(a/(1-a))
Eb_Pab.v = -np.log(b/(1-b)*(1-b)/(1-b-a))

# Pa_eq_1['g'] = (f_A['g']-Pab['g'])/(1+np.exp(Eb_Pa.v))
# Pb_eq_2['g'] = (f_B['g']-Pab['g'])/(1+np.exp(Eb_Pb.v))

# Pab_eq_3['g'] = 0.5*(1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B['g']-Pb['g'])))
# Pab_eq_4['g'] = 0.5*(1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A['g']-Pa['g'])))

Pa_eq_1['g'] = (f_A['g']-Pab['g'])/(1+np.exp(Eb_Pa.v))
Pb_eq_2['g'] = (f_B['g']-Pab['g'])/(1+np.exp(Eb_Pb.v))

Pab_eq_3['g'] = 0.5*(1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B['g']-Pb['g'])))
Pab_eq_4['g'] = 0.5*(1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A['g']-Pa['g'])))

Pab['g'] = 0.5*(1*Pab_eq_3['g']+1*Pab_eq_4['g'])


for i in range(1000):
    Pa_eq_1['g'] = (f_A['g']-Pab['g'])/(1+np.exp(Eb_Pa.v))
    Pb_eq_2['g'] = (f_B['g']-Pab['g'])/(1+np.exp(Eb_Pb.v))
    
    Pab_eq_3['g'] = 0.5*(1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B['g']-Pb['g'])))
    Pab_eq_4['g'] = 0.5*(1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A['g']-Pa['g'])))

    # Pab_eq_3['g'] = 0.5*(1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(1-Pb['g'])))*f_D['g']*f_B['g']
    # Pab_eq_4['g'] = 0.5*(1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(1-Pa['g'])))*f_D['g']*f_A['g']


    # Pa['g'] = Pa_eq_1['g']
    # Pb['g'] = Pb_eq_2['g']

    Pab['g'] = 0.5*(1*Pab_eq_3['g']+1*Pab_eq_4['g'])



Pab['g'] = 1*Pab['g']
Pa['g'] = 1*Pa['g']
Pb['g'] = 1*Pb['g']

#%%

# Pab['g'] = 0.585*f_D['g']
# Pa['g'] = 0.05*(f_A['g']-Pab['g'])
# Pb['g'] = 0.05*(f_B['g']-Pab['g'])

# Pab_eq_3['g'] = 0.585*f_D['g']
# Pab_eq_4['g'] = 0.585*f_D['g']

# Pa_eq_1['g'] = 0.05*(f_A['g']-Pab['g'])
# Pb_eq_2['g'] = 0.05*(f_B['g']-Pab['g'])

#%%
plt.figure(dpi=200)
plt.plot(x[0],f_A['g'],label = "$f^{A}$")
plt.plot(x[0],f_B['g'],label = "$f^{B}$")
plt.plot(x[0],Pab['g'],label = "$P^{ab}$")
plt.plot(x[0],Pa['g'],label = "$P^{a}$")
plt.plot(x[0],Pb['g'],label = "$P^{b}$")

plt.plot(x[0],Pa_eq_1['g'],label = "$P^{a}_{eq}$")
plt.plot(x[0],Pb_eq_2['g'],label = "$P^{b}_{eq}$")

plt.plot(x[0],Pab_eq_3['g'],label = "$P^{ab}_{eq}$")
plt.plot(x[0],Pab_eq_4['g'],label = "$P^{ab}_{eq}$")


# plt.plot(x[0],Pa['g']+Pb['g'])
# plt.hlines(0.6, -0.5, 0.5)
plt.ylim(-0.1,1.1)
plt.legend()
plt.show()



# %% EQUATIONS OF THE PROBLEM
# it's better to write the full equations at once instead of using variables

problem = d3.IVP([ f_A,f_B,f_D,
                    # Ma, Mb, Mab, Pab,
                    Pab, Pa,Pb,
                    Pa_eq_1, Pb_eq_2, Pab_eq_3, Pab_eq_4,
                    # C_Mab_Mab_inv,
                    # V_A,V_B,
                    Overlap,
                    F_A,F_B,
                    Coeff_fri,
                    grad_mu_fA,grad_mu_fB,
                    u_el,
                    F_fA_vis, F_fA_fri, F_fA_ent, F_fA_act, F_fA_ela,
                    F_fB_vis, F_fB_fri, F_fB_ent, F_fB_act, F_fB_ela],
                  namespace=locals()) # Declaration of the problem variables

# - Cahn Hillard equation for the filaments - #
problem.add_equation("dt(f_A) +D_f*ddx(-2*f_A +G_f*ddx(f_A)) = D_f*ddx(4*(f_A)**3-6*(f_A)**2) -dx(f_A*V_A) ")
problem.add_equation("dt(f_B) +D_f*ddx(-2*f_B +G_f*ddx(f_B)) = D_f*ddx(4*(f_B)**3-6*(f_B)**2) -dx(f_B*V_B)")
problem.add_equation("f_D = f_A*f_B")

#- Equation of the particles - #
problem.add_equation("dt(Pa)"
                      "-r*D_Pa.v*ddx(Pa)" # stabilization
                      "-D_Pa.v*ddx(Pa)" # diffusion 
                       "+Koff_1_Pa.v*Pa -Koff_3_Pab.v*Pab" # chemical reactions
                      "="
                      "-r*D_Pa.v*ddx(Pa)" # stabilization
                      "-D_Pa.v*dx(Pa/(h+f_A-Pab)*dx(f_A))" # 
                      "+D_Pa.v*dx(Pa/(h+f_A-Pab)*dx(Pab))" #
                      "-dx(Pa*V_A)" # convected flux
                        "+Koff_1_Pa.v*Pa_eq_1  -Koff_3_Pab.v*Pab_eq_3" # chemical reactions
                        )

problem.add_equation("dt(Pb)"
                      "-r*D_Pb.v*ddx(Pb)" # stabilization
                      "-D_Pb.v*ddx(Pb)" # diffusion 
                       "+Koff_2_Pb.v*Pb -Koff_4_Pab.v*Pab" # chemical reactions
                      "="
                      "-r*D_Pb.v*ddx(Pb)" # stabilization
                      "-D_Pb.v*dx(Pb/(h+f_B-Pab)*dx(f_B))" # 
                      "+D_Pb.v*dx(Pb/(h+f_B-Pab)*dx(Pab))" #
                      "-dx(Pb*V_B)" # convected flux
                       " +Koff_2_Pb.v*Pb_eq_2  -Koff_4_Pab.v*Pab_eq_4" # chemical reactions
                        )


problem.add_equation("dt(Pab)"
                      "-r*D_Pab.v*ddx(Pab)" # stabilization
                      "-D_Pab.v*ddx(Pab)" # diffusion 
                       "+Koff_3_Pab.v*Pab +Koff_4_Pab.v*Pab" # chemical reactions
                      "="
                      "-r*D_Pab.v*ddx(Pab)" # stabilization
                      "-D_Pab.v*dx( (f_B/(h+f_D-Pab) +1/(h+f_A-Pab)-1/(h+f_A-Pab-Pa))/( 1/(Pab)+1/(h+f_D-Pab) +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(f_A))" # 
                      "-D_Pab.v*dx( (f_A/(h+f_D-Pab) +1/(h+f_B-Pab)-1/(h+f_B-Pab-Pb))/( 1/(Pab)+1/(h+f_D-Pab) +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(f_B))" # 
                      "-D_Pab.v*dx( 1/(h+f_A-Pab-Pa)/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(Pa))" # 
                      "-D_Pab.v*dx( 1/(h+f_B-Pab-Pb)/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(Pb))" # 
                      "-dx(Pab*0.5*(V_A+V_B))" # convected flux
                       "+Koff_3_Pab.v*Pab_eq_3  +Koff_4_Pab.v*Pab_eq_4" # chemical reactions
                            )


# - Equation of the mobilities - #
# problem.add_equation("C_Mab_Mab_inv = 1/( 1/(Mab)+1/(h+f_D-Mab-Pab)  +1/(h+f_A-Mab-Pab-Ma)-1/(h+f_A-Mab-Pab) +1/(h+f_B-Mab-Pab-Mb)-1/(h+f_B-Mab-Pab) )")


# - Equation of the equilibrium concentrations - #
problem.add_equation("Pa_eq_1 = (h+f_A-Pab)/(1+np.exp(Eb_Pa.v))")
problem.add_equation("Pb_eq_2 = (h+f_B-Pab)/(1+np.exp(Eb_Pb.v))")


problem.add_equation("Pab_eq_3 = 0.5*(1+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B-Pb)))")
problem.add_equation("Pab_eq_4 = 0.5*(1+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A-Pa)))")

# problem.add_equation("Pa_eq_1 = 0.05*(f_A-Pab)")
# problem.add_equation("Pb_eq_2 = 0.05*(f_B-Pab)")

# problem.add_equation("Pab_eq_3 = 0.585*(h+f_D)")
# problem.add_equation("Pab_eq_4 = 0.585*(h+f_D)")

# problem.add_equation("Pab_eq_5 = (f_D-Mab)/(1+np.exp(Eb_Pab.v))")


# - Equation of elastic stress - #
problem.add_equation("dt(u_el) + G.v*u_el = K.v*(V_B-V_A)/Ly.v  ")


# - Gradient of the chemical potential of the filaments - #
problem.add_equation("grad_mu_fA = -1*( np.log((h_log+f_D)/(h_log+f_D-Pab))*dx(f_B) +f_B*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Pab)*dx(f_D-Pab)) + 1/(h+f_A-Pab)*dx(f_A-Pab) -1/(h+f_A-Pab-Pa)*dx(f_A-Pab-Pa)  )")
problem.add_equation("grad_mu_fB = -1*( np.log((h_log+f_D)/(h_log+f_D-Pab))*dx(f_A) +f_A*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Pab)*dx(f_D-Pab)) + 1/(h+f_B-Pab)*dx(f_B-Pab) -1/(h+f_B-Pab-Pb)*dx(f_B-Pab-Pb)  )")

# - Coefficients - #
problem.add_equation("Coeff_fri = Lz.v/Ly.v*eta.v*d3.Integrate((f_D )  ,('x'))")

# - Integration of the forces - #
problem.add_equation("F_fA_vis = -gamma.v*V_A  ")
problem.add_equation("F_fA_fri = Coeff_fri*(V_B-V_A)")
problem.add_equation("F_fA_ent = -n_s.v*d3.Integrate(f_A*( grad_mu_fA)  ,('x'))")
problem.add_equation("F_fA_act = -act.v*d3.Integrate(f_A*(Mab)  ,('x'))")
problem.add_equation("F_fA_ela = -E.v*Lz.v*d3.Integrate(f_A*(u_el)  ,('x'))")

problem.add_equation("F_fB_vis = -gamma.v*V_B  ")
problem.add_equation("F_fB_fri = -Coeff_fri*(V_B-V_A)")
problem.add_equation("F_fB_ent = -n_s.v*d3.Integrate(f_B*( grad_mu_fB)  ,('x'))")
problem.add_equation("F_fB_act =  act.v*d3.Integrate(f_B*(Mab)  ,('x'))")
problem.add_equation("F_fB_ela =  E.v*Lz.v*d3.Integrate(f_B*(u_el)  ,('x'))")


#In this experiment the A is fixed, and B is force-free.
# - Forces - #
problem.add_equation("F_A = F_fA_vis +F_fA_fri +F_fA_ent +F_fA_act +F_fA_ela") 
problem.add_equation("F_B = F_fB_vis +F_fB_fri +F_fB_ent +F_fB_act +F_fB_ela") 

# - Velocities - #
# problem.add_equation("V_B = v_B")
# problem.add_equation("V_A = v_A")

problem.add_equation("dt(Overlap)=V_A-V_B")


#%% SETTING THE VELOCITIES FOR THE SIMULATION





# ib = 550
# ie = 750

'Velocity from file'
folder_velocity = "velocity_profiles"
# name_file_velocity = "2210_VP_1_jump_s1" # 1jump
name_file_velocity = "1112_VP_4_jumps_s1" # 4jumps

file_ext = ".csv"
File_velocity = pd.read_csv(working_path +  '/'+folder_velocity + '/'+name_file_velocity+file_ext)
# data = np.transpose(np.array(File_velocity)[ib:ie])
data = np.transpose(np.array(File_velocity))




'Setting the arrays to set the velocity'


real_time = data[0]
stop_time = (real_time[-1]-real_time[0])/1000 # real time in ms
sim_time = np.linspace(0,stop_time,int(stop_time/timestep)) # time


v_A = np.zeros(len(sim_time))
v_B = np.zeros(len(sim_time))

pos_xl_A = np.zeros(len(sim_time))
pos_xr_A = np.zeros(len(sim_time))
pos_xl_B = np.zeros(len(sim_time))
pos_xr_B = np.zeros(len(sim_time))

#%%

mean_dt = (data[0][-1]-data[0][0])/len(data[0])

V_data = LS.function_convertion_array(data[1], len(sim_time))

v_A = -0.5*len(data[1])/(len(sim_time))/timestep*V_data*mean_dt
v_B = -1*v_A

vA = interp1d(sim_time, v_A, kind='linear')  # 'linear', 'quadratic', 'cubic', etc.
vB = interp1d(sim_time, v_B, kind='linear')  # 'linear', 'quadratic', 'cubic', etc.

Vmax = np.max(abs(v_A))



'Plotting'
pos_xl_A[0]=xl_A
pos_xr_A[0]=xr_A
pos_xl_B[0]=xl_B
pos_xr_B[0]=xr_B


for i in range(len(sim_time)-1):
    pos_xl_A[i+1]=pos_xl_A[i]+v_A[i]*timestep
    pos_xr_A[i+1]=pos_xr_A[i]+v_A[i]*timestep
    pos_xl_B[i+1]=pos_xl_B[i]+v_B[i]*timestep
    pos_xr_B[i+1]=pos_xr_B[i]+v_B[i]*timestep
   
    
    
# stop_time=1
(pos_xr_A[-1]-pos_xl_B[-1])-((pos_xr_A[0]-pos_xl_B[0]))
     
#%%   
    
# plt.figure(dpi=200)  
# plt.title(label="Position of filaments with respect to time")  
# plt.plot(pos_xl_A+li/2,sim_time,alpha=0.5,color="blue")
# plt.plot(pos_xl_A-li/2,sim_time,alpha=0.5,color="blue")
# plt.plot(pos_xl_A,sim_time,color="blue",label="A")
# plt.plot(pos_xr_A+li/2,sim_time,alpha=0.5,color="blue")
# plt.plot(pos_xr_A-li/2,sim_time,alpha=0.5,color="blue")
# plt.plot(pos_xr_A,sim_time,color="blue")
# plt.plot(pos_xl_B+li/2,sim_time,alpha=0.5,color="red")
# plt.plot(pos_xl_B-li/2,sim_time,alpha=0.5,color="red")
# plt.plot(pos_xl_B,sim_time,color="red",label="B")
# plt.plot(pos_xr_B+li/2,sim_time,alpha=0.5,color="red")
# plt.plot(pos_xr_B-li/2,sim_time,alpha=0.5,color="red")
# plt.plot(pos_xr_B,sim_time,color="red")
# plt.legend()
# plt.show()  

plt.figure(dpi=200)  
plt.title(label="Velocity of filaments with respect to time")  
plt.plot(v_A,sim_time,color="blue",label="A")
plt.plot(v_B,sim_time,color="red",label="B")
# plt.hlines(3.2, -0.05, 0.05)
plt.legend()
plt.show()  

#%%

hk=1e-3
plt.plot(sim_time, 2e-4*(-np.log((0.1+abs(vA(sim_time))/(Vmax+hk))/(1-abs(vA(sim_time))/(Vmax+hk)))*0.1+1))
plt.show()


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
    V_A['g'] = vA(solver.sim_time)
    V_B['g'] = vB(solver.sim_time)
    t=t+1   
    
    hk = 1e-3
    # timestep = 1e-4*(-np.log((0.1+abs(vA(solver.sim_time))/0.0864)/(1-abs(vA(solver.sim_time))/0.0864))*0.1+1)
    timestep = timestep0*(-np.log((0.1+abs(vA(solver.sim_time))/(Vmax+hk))/(1-abs(vA(solver.sim_time))/(Vmax+hk)))*0.1+1)
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
        print(F_fA_ent['g']*4e-3)
        
    
        if j%1  == 0 and j<10 or j%20 == 0:
                f_A.change_scales(1)
                f_B.change_scales(1)
                f_D.change_scales(1)
                
                Pab.change_scales(1)
                Pab_eq_3.change_scales(1)
                Pab_eq_4.change_scales(1)

                Pa.change_scales(1)
                Pa_eq_1.change_scales(1)
                
                Pb.change_scales(1)
                Pb_eq_2.change_scales(1)

                #Mb.change_scales(1)
                plt.figure(dpi=100)
                plt.title(V_B['g'])
                plt.plot(x[0],f_A['g'],color = 'blue',alpha = 0.5)
                plt.plot(x[0],f_B['g'],color = 'red',alpha = 0.5)
                
                plt.plot(x[0],Pab['g'],color = 'violet',label = "$P^{ab}$")
                plt.plot(x[0],Pab_eq_3['g'],color = 'violet',label = "$P^{ab}$",linestyle="--")
                plt.plot(x[0],Pab_eq_4['g'],color = 'violet',label = "$P^{ab}$",linestyle="--")
                

                plt.plot(x[0],Pa['g'],color = 'blue',label = "$P^{a}$")
                plt.plot(x[0],Pa_eq_1['g'],color = 'blue',label = "$P^{a}$",linestyle="--")
 
                
                plt.plot(x[0],Pb['g'],color = 'red',label = "$P^{b}$")
                plt.plot(x[0],Pb_eq_2['g'],color = 'red',label = "$P^{b}$",linestyle="--")


                # plt.plot(x[0],grad_mu_fB['g']/np.max(np.abs(grad_mu_fB['g'])),color = 'black',label = "$f$")
                # plt.ylim(-0.1,0.2)

                
                plt.legend()
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
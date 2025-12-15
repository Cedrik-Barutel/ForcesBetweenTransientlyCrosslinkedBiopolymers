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



#%%
V_list = [0.075,0.100,0.150,0.200,0.3,1]
V_list = [1]



# V_list = [0.025]#,0.050,0.075,0.100,0.150,0.200]

N_list = [5,10,20,30,40,50]
# N_list = [300]


Vname_list = ['0075','0100','0150','0200']
Vname_list = ['1']

# Vname_list = ['0025']

Nname_list = ['5','10','20','30','40','50']
# Nname_list = ['300']


# N_list = [5,10,20,30,40,50,60,70,80,]


# V_list = np.linspace(0.01, 1,9)
# V_list = [0.001,0.003,0.01,0.03,0.1,0.3,1]

N_list = np.linspace(1,201,9)
#%%

Vname_list = []
Nname_list = []

def float_to_digits(f): 
    return str(f).replace(".", "")#[:4]


def floor_to_decade(x):
    if x == 0:
        return 0.0
    
    s = f"{abs(x):.1e}"      # scientific notation string
    lead_str, exp_str = s.split("e")   # e.g. "3.474500000000000e-04"
    
    exp = int(exp_str)        # exponent
    lead = float(lead_str)    # leading digits
    
    # floor the leading digit
    lead_f = int(lead)        # works because int() floors positive numbers
    
    # reconstruct the value
    result = lead_f * (10 ** exp)
    
    return result if x >= 0 else -result
V_list  =  [floor_to_decade(v) for v in V_list]
Vname_list = [float_to_digits(v) for v in V_list]
Nname_list = [float_to_digits(n) for n in N_list]



print(Vname_list)
print(Nname_list)

#%%

# V_list = V_list[4:]
# Vname_list = Vname_list[4:]

# N_list = N_list[8:]
# Nname_list = Nname_list[8:]


#%%



for index_V in range(len(V_list)):
    for index_N in range(len(N_list)):
    
    # folder_name = "Gaska2020_2409_run_V200" # name of the simulation 
    
        folder_name = "tets_Gaska2020_1111_MAP_run_XXX_autoV"+str(Vname_list[index_V])+"N"+str(Nname_list[index_N]) # name of the simulation 
    
        working_path = os.path.dirname(__file__)
        folder = working_path  + "/"+ folder_name
        
        N_save = 500 # number of save
        timestep = 1e-2 #timestep
        stop_time = 100 # max simulating time 
        Lx = 20 #length of the simulation box
        Nx = 2**9
        
        
        'Dimensions of the filaments'
        # position of the filaments
        xl_A = -9
        # xr_A = 0.77
        # xl_B = -0.77
        xr_A = 0
        xl_B = -5
        # xl_B = -5
        # xl_B = -6
        
        xr_B = xl_B+9
        
        LA = xr_A-xl_A
        LB = xr_B-xl_B
        
        
        Ly = LS.Coefficient("Ly",0.02) # Distance between the filaments
        Lz = LS.Coefficient("Lz",0.02) # Witdh of the filaments
        
        n_s = LS.Coefficient("n_s",200) # Lineic binsind site concentration
        
        Eb_Pa = LS.Coefficient("Eb_Pa",0) # motors bound to (a)
        Eb_Pb = LS.Coefficient("Eb_Pb",0) # motors bound to (b)
        Eb_Pab = LS.Coefficient("Eb_Pab",-1) # passive bound to (ab) 
        
        
        D_Pa = LS.Coefficient("D_Pa",0.13) # motors bound to (a)
        D_Pb = LS.Coefficient("D_Pb",0.13) # motors bound to (b)
        D_Pab = LS.Coefficient("D_Pab",0.04) # passive bound to (ab) 
        
        
        
        Koff_5_Pa = LS.Coefficient("Koff_5_Pa",0.002) # Rates of reaction 1 around Ma_eq
        Koff_6_Pb = LS.Coefficient("Koff_6_Pb",0.002) # Rates of reaction 2 around Mb_eq
        Koff_7_Pab = LS.Coefficient("Koff_7_Pab",0.002) # Rates of reaction 3 around Mab_eq MEASURED
        Koff_8_Pab = LS.Coefficient("Koff_8_Pab",0.002) # Rates of reaction 4 around Mab_eq MEASURED
        # Koff_7_Pab = LS.Coefficient("Koff_7_Pab",2) # Rates of reaction 3 around Mab_eq MEASURED
        # Koff_8_Pab = LS.Coefficient("Koff_8_Pab",2) # Rates of reaction 4 around Mab_eq MEASURED
        
        
        eta = LS.Coefficient("eta",500) # Shear viscosity
        gamma = LS.Coefficient("gamma",100) # Viscous friction
        E = LS.Coefficient("E",0) # Shear elasticity modulus
        # act = LS.Coefficient("act",288) # Activity coefficient constant
        act = LS.Coefficient("act",0) # Activity coefficient withmotors
        
        K = LS.Coefficient("K",0) # Convertion bwetween elastic strain and velocity
        G = LS.Coefficient("G",0) # Elastic relaxation
        
        
        v_B = LS.Coefficient("v_B",V_list[index_V]) # Rates of reaction 1 around Ma_eq
        
        stop_time = 5.5/v_B.v # max simulating time 
        while stop_time/v_B.v/500/timestep <= 1:
            timestep = timestep*1e-1
            
        # print(stop_time)
        # print(timestep)
        
        # print()
        # print(N_list[index_N]/(n_s.v*(xr_A-xl_B)))
        
        #%%
        h = 2e-3 # numerical help to avoid division by 0
        h_log = 2*h # numerical help to log(0)
        
        r = 0  # Stabilization with r*diffusion coefficient
        
        # Phase field model 
        li = 0.2 # length of the filaments interface
        D_f = 0.4 # diffusion coefficient of the phase field
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
        Pab = dist.Field(name='Pab',bases=(xbasis)) # Concentration of 
        
        
        # Equilibirum densities
        
        Pa_eq_5 = dist.Field(name='Pa_eq_5',bases=(xbasis)) # Equilibrium Concentration of Ma 
        Pb_eq_6 = dist.Field(name='Pb_eq_6',bases=(xbasis)) # Concentration of 
        Pab_eq_7 = dist.Field(name='Pab_eq_7',bases=(xbasis)) # Concentration of
        Pab_eq_8 = dist.Field(name='Pab_eq_8',bases=(xbasis)) # Concentration of  
         
        
        
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
        
        
        F_A = dist.Field(name='F_A')
        F_B = dist.Field(name='F_B')
        
        
        #%% INITIAL CONDITIONS
        'FILAMENTS PHASE FIELD'
        f_A['g'] = LS.function_filament(f_A['g'], xl_A, xr_A, x[0], li)
        f_B['g'] = LS.function_filament(f_B['g'], xl_B, xr_B, x[0], li)
        f_D['g'] = np.minimum(f_A['g'],f_B['g'])
        
        'NUMBER OF PARTICLE'
        # 1st guess on the number of particle definition
        
        h_log_d = 1e-5
        
        n_ma = 0.05
        n_mb = 0.05
        n_mab = 0.05
        
        n_pa = 0.01
        n_pb = 0.01
        n_pab = 0.05
        n_pab = N_list[index_N]/(n_s.v*(xr_A-xl_B))
        n_pa = n_pab/5
        n_pb = n_pab/5
        
        
        Pab['g'] = n_pab*f_D['g']
        
        
        Pa['g'] = n_pa*(f_A['g']-Pab['g'])
        Pb['g'] = n_pb*(f_B['g']-Pab['g'])
        
        Eb_Pa.v = -np.log(h_log_d+n_pa/(1-n_pa-n_ma))
        Eb_Pb.v = -np.log(h_log_d+n_pb/(1-n_pb-n_mb))
        
        Eb_Pab.v = -np.log(h_log_d+n_pab/(1-n_pab-n_mab-n_pa*(1-n_mab-n_pab))*(1-n_pab-n_mab)/(1-n_pab-n_mab-n_pa*(1-n_mab-n_pab)))
        
        #%%
        
        Pa_eq_5['g'] = (f_A['g']-Pab['g'])/(1+np.exp(Eb_Pa.v))
        Pb_eq_6['g'] = (f_B['g']-Pab['g'])/(1+np.exp(Eb_Pb.v))
        Pab_eq_7['g'] = 0.5*(1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B['g']-Pb['g'])))
        Pab_eq_8['g'] = 0.5*(1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A['g']-Pa['g'])))
        
        
        
        #%%
        for i in range(200):    
            
            Pab_eq_7['g'] = 0.5*(1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa['g']*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B['g']-Pb['g'])))
            Pab_eq_8['g'] = 0.5*(1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb['g']*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A['g']-Pa['g'])))
        
        
            Pab_eq_7['g'][Pab_eq_7['g']<=0] =0
            Pab_eq_8['g'][Pab_eq_7['g']<=0] =0
        
        
            Pab['g'] = 0.5*(Pab_eq_7['g']+Pab_eq_8['g'])
        
        
        
        
        #%%
        Pab['g']=0.3*f_D['g']-0.05*(x[0]-(xl_B+xr_B)/2)*f_D['g']
        # Mab['g']=0.0*f_D['g']
        Pa['g']=0.0*Pa['g']
        Pb['g']=0.0*Pb['g']
        Pab['g']=0.7*Pab['g']
        
        
        # %%
        # plt.figure(dpi=200)
        # plt.title("Passive")
        # plt.plot(x[0],f_A['g'],label = "$f^{A}$")
        # plt.plot(x[0],f_B['g'],label = "$f^{B}$")
        # plt.plot(x[0],Pa['g'],label = "$P^{a}$")
        # plt.plot(x[0],Pb['g'],label = "$P^{b}$")
        # plt.plot(x[0],Pab['g'],label = "$P^{ab}$")
        # plt.plot(x[0],0.3*f_D['g'],label = "$test$")
        
        # plt.plot(x[0],Pa_eq_5['g'],label = "$P^{a}$",alpha = 0.3)
        # plt.plot(x[0],Pb_eq_6['g'],label = "$P^{b}$",alpha = 0.3)
        # plt.plot(x[0],Pab_eq_7['g'],label = "$P^{ab}$",alpha = 0.3)
        # plt.plot(x[0],Pab_eq_8['g'],label = "$P^{ab}$",alpha = 0.3)
        
        # plt.ylim(-0.1,1.1)
        # plt.legend()
        # plt.show()
        
        
        print("N=" +str(n_s.v*integrate.simpson(Pab['g'],x[0])))

        
        
        # %% EQUATIONS OF THE PROBLEM
        # it's better to write the full equations at once instead of using variables
        
        problem = d3.IVP([ f_A,f_B,f_D,
                            Pa, Pb, Pab,
                            Pa_eq_5, Pb_eq_6, Pab_eq_7, Pab_eq_8,
                            # V_A,V_B,
                            F_A,F_B,
                            Coeff_fri,
                            grad_mu_fA,grad_mu_fB,
                            F_fA_vis, F_fA_fri, F_fA_ent, F_fA_ela,
                            F_fB_vis, F_fB_fri, F_fB_ent, F_fB_ela],
                          namespace=locals()) # Declaration of the problem variables
        
        # - Cahn Hillard equation for the filaments - #
        problem.add_equation("dt(f_A) +D_f*ddx(-2*f_A +G_f*ddx(f_A)) = D_f*ddx(4*(f_A)**3-6*(f_A)**2) -dx(f_A*V_A) ")
        problem.add_equation("dt(f_B) +D_f*ddx(-2*f_B +G_f*ddx(f_B)) = D_f*ddx(4*(f_B)**3-6*(f_B)**2) -dx(f_B*V_B)")
        problem.add_equation("f_D = f_A*f_B")
        
        #- Equation of the particles - #
        
        
        
        problem.add_equation("dt(Pa)"
                             "-r*D_Pa.v*ddx(Pa)" # stabilization
                             "-D_Pa.v*ddx(Pa)" # diffusion 
                             "+Koff_5_Pa.v*Pa -Koff_7_Pab.v*Pab" # chemical reactions
                             "="
                             "-r*D_Pa.v*ddx(Pa)" # stabilization
                             "-D_Pa.v*dx(Pa/(h+f_A-Pab)*dx(f_A))" # 
                             "+D_Pa.v*dx(Pa/(h+f_A-Pab)*dx(Pab))" #
                             "-dx(Pa*V_A)" # convected flux
                             "+Koff_5_Pa.v*Pa_eq_5  -Koff_7_Pab.v*Pab_eq_7") # chemical reactions
        
        problem.add_equation("dt(Pb)"
                             "-r*D_Pb.v*ddx(Pb)" # stabilization
                             "-D_Pb.v*ddx(Pb)" # diffusion 
                             "+Koff_6_Pb.v*Pb -Koff_8_Pab.v*Pab" # chemical reactions
                             "="
                             "-r*D_Pb.v*ddx(Pb)" # stabilization
                             "-D_Pb.v*dx(Pb/(h+f_B-Pab)*dx(f_B))" # 
                             "+D_Pb.v*dx(Pb/(h+f_B-Pab)*dx(Pab))" #
                             "-dx(Pb*V_B)" # convected flux
                             "+Koff_6_Pb.v*Pb_eq_6  -Koff_8_Pab.v*Pab_eq_8") # chemical reactions
        
        
        problem.add_equation("dt(Pab)"
                              "-r*D_Pab.v*ddx(Pab)" # stabilization
                              "-D_Pab.v*ddx(Pab)" # diffusion 
                              "+Koff_7_Pab.v*Pab +Koff_8_Pab.v*Pab" # chemical reactions
                              "="
                              "-r*D_Pab.v*ddx(Pab)" # stabilization
                              "-D_Pab.v*dx( (f_B/(h+f_D-Pab) +1/(h+f_A-Pab)-1/(h+f_A-Pab-Pa))/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(f_A))" # 
                              "-D_Pab.v*dx( (f_A/(h+f_D-Pab) +1/(h+f_B-Pab)-1/(h+f_B-Pab-Pb))/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(f_B))" # 
                              "-D_Pab.v*dx( 1/(h+f_A-Pab-Pa)/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(Pa))" # 
                              "-D_Pab.v*dx( 1/(h+f_B-Pab-Pb)/( 1/(Pab)+1/(h+f_D-Pab)  +1/(h+f_A-Pab-Pa)-1/(h+f_A-Pab) +1/(h+f_B-Pab-Pb)-1/(h+f_B-Pab) )*dx(Pb))" # 
                              "-dx(Pab*0.5*(V_A+V_B))" # convected flux
                              "+Koff_7_Pab.v*Pab_eq_7  +Koff_8_Pab.v*Pab_eq_8") # chemical reactions
        
        
        
        # - Equation of the mobilities - #
        
        
        # - Equation of the equilibrium concentrations - #
        
        problem.add_equation("Pa_eq_5 = (f_A-Pab)/(1+np.exp(Eb_Pa.v))")
        problem.add_equation("Pb_eq_6 = (f_B-Pab)/(1+np.exp(Eb_Pb.v))")
        
        problem.add_equation("Pab_eq_7 = 0.5*(1+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))-np.sqrt( (1+Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v)))**2 -4*Pa*np.exp(-(Eb_Pab.v-Eb_Pa.v))*(f_B-Pb)))")
        problem.add_equation("Pab_eq_8 = 0.5*(1+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))-np.sqrt( (1+Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v)))**2 -4*Pb*np.exp(-(Eb_Pab.v-Eb_Pb.v))*(f_A-Pa)))")
        
        
        # - Equation of elastic stress - #
        
        
        # - Gradient of the chemical potential of the filaments - #
        problem.add_equation("grad_mu_fA = -1*( np.log((h_log+f_D)/(h_log+f_D-Pab))*dx(f_B) +f_B*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Pab)*dx(f_D-Pab)) + 1/(h+f_A-Pab)*dx(f_A-Pab) -1/(h+f_A-Pab-Pa)*dx(f_A-Pab-Pa)  )")
        problem.add_equation("grad_mu_fB = -1*( np.log((h_log+f_D)/(h_log+f_D-Pab))*dx(f_A) +f_A*( (1/(h+f_D))*dx(f_D) -1/(h+f_D-Pab)*dx(f_D-Pab)) + 1/(h+f_B-Pab)*dx(f_B-Pab) -1/(h+f_B-Pab-Pb)*dx(f_B-Pab)  )")
        
        # - Coefficients - #
        problem.add_equation("Coeff_fri = eta.v*d3.Integrate((n_s.v*(Pab) )  ,('x'))")
        
        # - Integration of the forces - #
        problem.add_equation("F_fA_vis = -gamma.v*V_A  ")
        problem.add_equation("F_fA_fri =  Coeff_fri*(V_B-V_A)/Ly.v*Lz.v")
        problem.add_equation("F_fA_ent = -n_s.v*d3.Integrate(f_A*( grad_mu_fA)  ,('x'))")
        problem.add_equation("F_fA_ela = E.v*Lz.v*d3.Integrate(f_A*(u_el*f_D)  ,('x'))")
        
        problem.add_equation("F_fB_vis = -gamma.v*V_B  ")
        problem.add_equation("F_fB_fri = -Coeff_fri*(V_B-V_A)/Ly.v*Lz.v")
        problem.add_equation("F_fB_ent = -n_s.v*d3.Integrate(f_B*( grad_mu_fB)  ,('x'))")
        problem.add_equation("F_fB_ela = -E.v*Lz.v*d3.Integrate(f_B*(u_el*f_D)  ,('x'))")
        
        
        #In this experiment the A is fixed, and B is force-free.
        # - Forces - #
        problem.add_equation("F_A = 0") 
        problem.add_equation("F_B = F_fB_vis +F_fB_fri +F_fB_ent +F_fB_ela") 
        
        # - Velocities - #
        # problem.add_equation("V_B = v_B.v")
        # problem.add_equation("V_A = 0")
        
        
        #%%
        u_el['g'] = 0
        
        sim_time = np.linspace(0,stop_time,int(stop_time/timestep)+50) # time
        
        array_v_B = np.zeros(len(sim_time))
        
        pos_xl_A = np.zeros(len(sim_time))
        pos_xr_A = np.zeros(len(sim_time))
        pos_xl_B = np.zeros(len(sim_time))
        pos_xr_B = np.zeros(len(sim_time))
        
        #%%
        
        
        # array_v_B = v_B.v*(LS.function_filament(v_B, 0, 20, sim_time, 1) + LS.function_filament(v_B, 40, 60, sim_time, 1))
        array_v_B = v_B.v*(np.ones(len(sim_time)))
        
        
        
        
        'Plotting'
        pos_xl_A[0]=xl_A
        pos_xr_A[0]=xr_A
        pos_xl_B[0]=xl_B
        pos_xr_B[0]=xr_B
        
        
        for i in range(len(sim_time)-1):
        
            pos_xl_B[i+1]=pos_xl_B[i]+array_v_B[i]*timestep
            pos_xr_B[i+1]=pos_xr_B[i]+array_v_B[i]*timestep
           
            
            
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
        plt.plot(array_v_B,sim_time,color="red",label="B")
        # plt.hlines(3.2, -0.05, 0.05)
        plt.legend()
        plt.show()  
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
            V_B['g'] = array_v_B[t]
        
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
                        f_D.change_scales(1)
                        
                        grad_mu_fB.change_scales(1)
                        Pa.change_scales(1)
                        Pb.change_scales(1)
                        Pab.change_scales(1)
        
        
                        plt.figure()
                        plt.title("V="+str(V_list[index_V])+" N="+str(N_list[index_N]))
                        plt.plot(x[0],f_A['g'],color = 'blue',alpha = 0.5)
                        plt.plot(x[0],f_B['g'],color = 'red',alpha = 0.5)
                        plt.plot(x[0],Pa['g'],color = 'blue',label = "$P^{a}$")
                        plt.plot(x[0],Pb['g'],color = 'red',label = "$P^{b}$")
                        plt.plot(x[0],Pab['g'],color = 'violet',label = "$P^{ab}$")
        
                        
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
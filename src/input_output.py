import common as g
import equations as eq
import numpy as np
import sys
import pickle

#------------------------------------------
# Read user input parameters
def read_input_parameters():

    g.fparam_path = sys.argv[1]
    print('Reading parameters from: {:s}'.format(g.fparam_path)) 

    f = open(g.fparam_path,'r')
    
    lines = f.readlines()
    # remove blank lines and labels
    lines = [l for l in lines if l.strip()]
    lines = [l for l in lines if l[0] != '#']

    # Read fluid properties
    li = 0
    g.mu, g.gamma, g.Pr, g.R_g, g.k, = \
        map(float,lines[li].split(','))
    # Read domain specification
    li = 1
    g.Lx, g.Ly, g.Lz = map(float,lines[li].split(','))
    # Read inlet conditions
    li = 2
    g.jet_height_y, g.jet_height_z, g.U_jet, g.V_jet, g.W_jet, g.P_jet, g.T_jet = \
        map(float,lines[li].split(','))
    
    # Read ambient conditions
    li = 3
    g.U_inf, g.V_inf, g.W_inf, g.P_inf, g.T_inf = \
        map(float,lines[li].split(','))

    # Read grid parameters
    li = 4
    g.nx, g.ny, g.nz = map(int,lines[li].split(','))
    
    # Read timestep parameters
    li = 5
    g.CFL_ref, = map(float,lines[li].split(','))
    li = 6
    g.nsteps, g.nsave, g.nmonitor = \
        map(int,lines[li].split(','))
    
    # Read input/output parameters
    li = 7
    g.fin_path = lines[li].replace('\'','').strip()
    li = 8
    g.fout_path = lines[li].replace('\'','').strip()
   
    print('Input data file: {:s}'.format(g.fin_path))
    print('Output data file: {:s}'.format(g.fout_path))
 
    # Close parameter input file
    f.close() 
    
    # Broadcast user input (mpi)
    

#-------------------------------------------
# Initialize flow field
def init_flow():
    
    if g.fin_path:
        # Use input data file
        read_input_data()
    else:
        # Default flow field initialization
        g.Rho[:,:,:] = g.Rho_inf
        g.U[:,:,:] = 0
        g.V[:,:,:] = 0
        g.W[:,:,:] = 0
        g.P[:,:,:] = g.P_inf
        Rho_,RhoU_,RhoV_,RhoW_,E_ = eq.PrimToCons(g.Rho,g.U,g.V,g.W,g.P)
        g.Q[:,:,:,0] = Rho_
        g.Q[:,:,:,1] = RhoU_
        g.Q[:,:,:,2] = RhoV_
        g.Q[:,:,:,3] = RhoW_
        g.Q[:,:,:,4] = E_

#-------------------------------------------
# Read data from input file
def read_input_data():
   raise Exception("Not implemented") 
    


#-------------------------------------------
# Write data to output file
def output_data():
    
    # Specify the output file
    fout_path = g.fout_path + '.' + str(g.tstep)

    saveVars = [g.xg,g.yg,g.zg,g.Q]

    with open(fout_path,'wb') as f:
        pickle.dump(saveVars,f)

     

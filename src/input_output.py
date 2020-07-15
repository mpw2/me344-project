import common as g
import sys

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
    g.mu, g.gamma, g.Pr, g.R_g, g.k, \
        g.Rho_ref, g.P_ref = map(float,lines[li].split(','))
    # Read domain specification
    li = 1
    g.Lx, g.Ly = map(float,lines[li].split(','))
    # Read inlet conditions
    li = 2
    g.Ujet, g.jet_height, g.Pjet = \
        map(float,lines[li].split(','))
    
    # Read grid parameters
    li = 3
    g.nx, g.ny = map(int,lines[li].split(','))
    
    # Read timestep parameters
    li = 4
    g.CFL_ref, = map(float,lines[li].split(','))
    li = 5
    g.nsteps, g.nsave, g.nmonitor = \
        map(int,lines[li].split(','))
    
    # Read input/output parameters
    li = 6
    g.fin_path = lines[li]
    li = 7
    g.fout_path = lines[li] 
   
    
    # Close parameter input file
    f.close() 
    
    # Broadcast user input (mpi)
    

#-------------------------------------------
# Initialize flow field
def init_flow():
    
    if fin_path is None:
        # Default flow field initialization
        Rho[:,:] = Rho_ref
        U[:,:] = 0
        V[:,:] = 0
        P[:,:] = Patm 
    else:
        # Use input data file
        read_input_data()

#-------------------------------------------
# Read data from input file
def read_input_data():
   raise Exception("Not implemented") 
    
    
    
    



#-------------------------------------------
# Write data to output file
def output_data():
    
    # Open the output file
    f = open(fout_path,"wb")
    
    arr=bytearray(nx,ny)
    
    ConsToPrim(Q,gamma)
    arr=bytearray()
    f.write(arr)

    # Close the output file
    f.close()
    
    
    
    
    
    
    
    
    

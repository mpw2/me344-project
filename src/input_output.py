from common import *

#------------------------------------------
# Read user input parameters
def read_input_parameters():
    global mu, gamma, Pr, R_g, k, Rho_ref, P_ref
    global Lx, Ly
    global nx, ny
    global Ujet, jet_height, Pjet
    global CFL_ref, nsteps, nsave, nmonitor
    global fin_path, fout_path, fparam_path

    fparam_path = argv[1]
    print('Reading parameters from: %s'.format(fparam_path)) 

    f = open(fparam_path,'r')
    
    lines = f.readlines()
    # remove blank lines and labels
    lines = [l for l in lines if l.strip()]
    lines = [l for l in lines if l[0] != '#']

    # Read fluid properties
    li = 0
    mu, gamma, Pr, R_g, k, Rho_ref, P_ref = map(float,lines[li].split(','))
    # Read domain specification
    li = 1
    Lx, Ly = map(float,lines[li].split(','))
    # Read grid parameters
    li = 2
    nx, ny = map(int,lines[li].split(','))
    # Read inlet conditions
    li = 3
    Ujet, jet_height, Pjet = map(float,lines[li].split(','))
    
    # Read timestep parameters
    li = 4
    CFL_ref = map(float,lines[li].split(','))
    li = 5
    nsteps, nsave, nmonitor = map(int,lines[li].split(','))
    
    # Read input/output parameters
    li = 6
    fin_path = lines[li]
    li = 7
    fout_path = lines[li] 
   
    
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
    
    
    
    
    
    
    
    
    

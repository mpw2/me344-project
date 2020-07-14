import common

#------------------------------------------
# 
def read_input_parameters():
    # Read in from user input file
    
    
    
    
    
    
    
    # Broadcast user input


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
    
    
    
    
    
    
    
    
    

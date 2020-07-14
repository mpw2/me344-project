#!/usr/bin/env python

#------------------------------------------------
#
# Finite Volume Jet Code
#
# Authors: Carlos Gonzalez, Michael Whitmore
#
# ... Solver details
#
#------------------------------------------------

def main():
   # Read user input file
   read_input()
   
   # Initialize data structures
   initialize()
   
   # Start time integration loop
   for n in range(Nt):
        
        # compute timestep size
        compute_dt()
        
        # time step
        compute_timestep_maccormack()
        
        
        # output to monitor
        if ( n % nmonitor == 0 ):
            output_monitor()
        
        # output to data file
        if ( n % nsave == 0 ):
            output_data()
    
    

if __name__ == "__main__":
    main()


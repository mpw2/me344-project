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

# Import jet code modules
from boundary_conditions        import *
from equations                  import *
from initialization             import *
from input_output               import *
from main                       import *
from monitor                    import *
from mpi                        import *
from time_integration           import *
import common as g

def main():
   # Read user input file
   read_input_parameters()
   
   # Initialize data structures
   initialize()
   
   # Start time integration loop
   for n in range(g.nsteps):
        
        # compute timestep size
        compute_dt()
        
        # time step
        compute_timestep_maccormack()
        
        
        # output to monitor
        if ( n % g.nmonitor == 0 ):
            output_monitor()
        
        # output to data file
        if ( n % g.nsave == 0 ):
            output_data()
    
    

if __name__ == "__main__":
    main()


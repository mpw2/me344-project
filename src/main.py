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

   output_monitor()
 
   # Start time integration loop
   for tstep in range(1,g.nsteps+1):
        g.tstep = tstep

        # compute timestep size
        compute_dt()
        
        # time step
        compute_timestep_maccormack()
        
        
        # output to monitor
        if ( tstep % g.nmonitor == 0 ):
            output_monitor()
        
        # output to data file
        if ( tstep % g.nsave == 0 ):
            output_data()
    
    

if __name__ == "__main__":
    main()


#!/usr/bin/env python

# ------------------------------------------------
#
# Finite Volume Jet Code
#
# Authors: Carlos Gonzalez, Michael Whitmore
#
# ... Solver details
#
# ------------------------------------------------

# Import jet code modules
import initialization as ini
import input_output as io
import monitor as mon
import time_integration as ti
import common as g


def main():
    # Read user input file
    io.read_input_parameters()

    # Initialize data structures
    ini.initialize()

    mon.output_monitor()

    # Start time integration loop
    for tstep in range(1, g.nsteps+1):
        g.tstep = tstep

        # compute timestep size
        ti.compute_dt()

        # time step
        ti.compute_timestep_maccormack()

        # output to monitor
        if (tstep % g.nmonitor == 0):
            mon.output_monitor()

        # output to data file
        if (tstep % g.nsave == 0):
            io.output_data()


if __name__ == "__main__":
    main()


#

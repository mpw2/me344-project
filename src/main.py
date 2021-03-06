#!/usr/bin/env python
"""
Finite Difference Jet Code

Authors: Carlos Gonzalez, Michael Whitmore

Solver Details:
    - 2nd Order Finite Difference
    - MacCormack Scheme

Package Requirements:
    Solver:
        - numpy
        - mpi4py
        - numba
    Post-processing:
        - matplotlib
"""
import sys

import initialization as ini
import input_output as io
import monitor as mon
import time_integration as ti
import common as g
import time


def main():
    """main method: run time integration loop"""

    # initialize data structures
    start = time.time()
    ini.initialize()

    # output initial condition to monitor
    mon.output_monitor()

    # start time integration loop
    for tstep in range(1, g.nsteps+1):
        g.tstep = tstep

        # compute timestep size
        ti.compute_dt()

        # time step
        ti.compute_timestep_maccormack()

        # output to monitor
        if tstep % g.nmonitor == 0:
            mon.output_monitor()

        # output to data file
        if tstep % g.nsave == 0:
            io.output_data()

    # finish up
    mon.output_final()
    end = time.time()
    if g.myrank == 0:
        print("Elapsed time = " + str(end - start))
        sys.stdout.flush()


if __name__ == "__main__":
    main()


#

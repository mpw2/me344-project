"""input_output.py

Description:
    Module for reading input and output data from files. Includes
    reading parameter input files, reading initial condition data files,
    and writing solution data files.

Dependencies:
    sys - get command line argument with parameters filepath
    pickle - format for reading and writing solution data
"""

import sys
import pickle

import common as g
import equations as eq
import mpi


def read_input_parameters():
    """Read user input parameters from file.

    Parameter file specified in argv[1]
    """
    if mpi.myrank == 0:
        # get parameter input file
        g.fparam_path = sys.argv[1]
        print('Reading parameters from: {:s}'.format(g.fparam_path))

        f = open(g.fparam_path, 'r')

        lines = f.readlines()
        # remove blank lines and comments/labels
        lines = [ln for ln in lines if ln.strip()]
        lines = [ln for ln in lines if ln[0] != '#']

        # Read fluid properties
        li = 0
        g.mu, g.gamma, g.Pr, g.R_g, g.k, g.D = \
            map(float, lines[li].split(', '))
        # Read domain specification
        li = 1
        g.Lx, g.Ly, g.Lz = map(float, lines[li].split(', '))
        # Read inlet conditions
        li = 2
        g.jet_height_y, g.jet_height_z, g.U_jet, g.V_jet, g.W_jet, \
            g.P_jet, g.T_jet, g.Phi_jet = \
            map(float, lines[li].split(', '))

        # Read ambient conditions
        li = 3
        g.U_inf, g.V_inf, g.W_inf, g.P_inf, g.T_inf, g.Phi_inf = \
            map(float, lines[li].split(', '))

        # Read grid parameters
        li = 4
        g.nx, g.ny, g.nz = map(int, lines[li].split(', '))

        # Read timestep parameters
        li = 5
        g.CFL_ref, = map(float, lines[li].split(', '))
        li = 6
        g.nsteps, g.nsave, g.nmonitor = \
            map(int, lines[li].split(', '))

        # Read input/output parameters
        li = 7
        g.fin_path = lines[li].replace('\'', '').strip()
        li = 8
        g.fout_path = lines[li].replace('\'', '').strip()

        print('Input data file: {:s}'.format(g.fin_path))
        print('Output data file: {:s}'.format(g.fout_path))

        # Close parameter input file
        f.close()

    # Broadcast user input (mpi)
    # fluid properties
    g.mu = mpi.comm.bcast(g.mu, root=0)
    g.gamma = mpi.comm.bcast(g.gamma, root=0)
    g.Pr = mpi.comm.bcast(g.Pr, root=0)
    g.R_g = mpi.comm.bcast(g.R_g, root=0)
    g.k = mpi.comm.bcast(g.k, root=0)
    g.D = mpi.comm.bcast(g.D, root=0)
    # domain specification
    g.Lx = mpi.comm.bcast(g.Lx, root=0)
    g.Ly = mpi.comm.bcast(g.Ly, root=0)
    g.Lz = mpi.comm.bcast(g.Lz, root=0)
    # inlet conditions
    g.jet_height_y = mpi.comm.bcast(g.jet_height_y, root=0)
    g.jet_height_z = mpi.comm.bcast(g.jet_height_z, root=0)
    g.U_jet = mpi.comm.bcast(g.U_jet, root=0)
    g.V_jet = mpi.comm.bcast(g.V_jet, root=0)
    g.W_jet = mpi.comm.bcast(g.W_jet, root=0)
    g.P_jet = mpi.comm.bcast(g.P_jet, root=0)
    g.T_jet = mpi.comm.bcast(g.T_jet, root=0)
    g.Phi_jet = mpi.comm.bcast(g.Phi_jet, root=0)
    # ambient conditions
    g.U_inf = mpi.comm.bcast(g.U_inf, root=0)
    g.V_inf = mpi.comm.bcast(g.V_inf, root=0)
    g.W_inf = mpi.comm.bcast(g.W_inf, root=0)
    g.P_inf = mpi.comm.bcast(g.P_inf, root=0)
    g.T_inf = mpi.comm.bcast(g.T_inf, root=0)
    g.Phi_inf = mpi.comm.bcast(g.Phi_inf, root=0)
    # grid parameters
    g.nx = mpi.comm.bcast(g.nx, root=0)
    g.ny = mpi.comm.bcast(g.ny, root=0)
    g.nz = mpi.comm.bcast(g.nz, root=0)
    # timestep parameters
    g.CFL_ref = mpi.comm.bcast(g.CFL_ref, root=0)
    g.nsteps = mpi.comm.bcast(g.nsteps, root=0)
    g.nsave = mpi.comm.bcast(g.nsave, root=0)
    g.nmonitor = mpi.comm.bcast(g.nmonitor, root=0)
    # input/output parameters
    g.fin_path = mpi.comm.bcast(g.fin_path, root=0)
    g.fout_path = mpi.comm.bcast(g.fout_path, root=0)


def init_flow():
    """Initialize the flow field.

    If fin_path is specified, initialize with a data file.
    Else use the default initial condition.
    """
    if g.fin_path:
        # Use input data file
        read_input_data()
    else:
        # Default flow field initialization
        g.Rho[:, :, :] = g.Rho_inf
        g.U[:, :, :] = 0
        g.V[:, :, :] = 0
        g.W[:, :, :] = 0
        g.P[:, :, :] = g.P_inf
        g.Phi[:, :, :] = 0
        Rho_, RhoU_, RhoV_, RhoW_, E_, RhoPhi_ = \
            eq.PrimToCons(g.Rho, g.U, g.V, g.W, g.P, g.Phi)
        g.Q[:, :, :, 0] = Rho_
        g.Q[:, :, :, 1] = RhoU_
        g.Q[:, :, :, 2] = RhoV_
        g.Q[:, :, :, 3] = RhoW_
        g.Q[:, :, :, 4] = E_
        g.Q[:, :, :, 5] = RhoPhi_


def read_input_data():
    """Read flow variables from data file fin_path"""
    raise Exception("Not implemented")


def output_data():
    """Write flow state to output file fout_path"""
    if mpi.myrank == 0:
        # Specify the output file
        fout_path = g.fout_path + '.' + str(g.tstep)

        save_vars = [g.xg, g.yg, g.zg, g.Q]

        f = open(fout_path, 'wb')
        pickle.dump(save_vars, f)
        f.close()


#

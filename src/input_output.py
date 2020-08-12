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
import numpy as np

import common as g
import equations as eq


def read_input_parameters():
    """Read user input parameters from file.

    Parameter file specified in argv[1]
    """
    if g.myrank == 0:
        # get parameter input file
        g.fparam_path = sys.argv[1]
        print('Reading parameters from: {:s}'.format(g.fparam_path))
        sys.stdout.flush()

        fil = open(g.fparam_path, 'r')

        lines = fil.readlines()
        # remove blank lines and comments/labels
        lines = [ln for ln in lines if ln.strip()]
        lines = [ln for ln in lines if ln[0] != '#']

        # Read fluid properties
        lix = 0
        g.mu, g.gamma, g.Pr, g.R_g, g.k, g.D = \
            map(np.float64, lines[lix].split(', '))
        # Read domain specification
        lix = 1
        g.Lx, g.Ly, g.Lz = map(np.float64, lines[lix].split(', '))
        # Read inlet conditions
        lix = 2
        g.jet_height_y, g.jet_height_z, g.U_jet, g.V_jet, g.W_jet, \
            g.P_jet, g.T_jet, g.Phi_jet = \
            map(np.float64, lines[lix].split(', '))

        # Read ambient conditions
        lix = 3
        g.U_inf, g.V_inf, g.W_inf, g.P_inf, g.T_inf, g.Phi_inf = \
            map(np.float64, lines[lix].split(', '))

        # Read grid parameters
        lix = 4
        g.nx, g.ny, g.nz = map(np.int32, lines[lix].split(', '))

        # Read timestep parameters
        lix = 5
        g.CFL_ref, = map(np.float64, lines[lix].split(', '))
        lix = 6
        g.nsteps, g.nsave, g.nmonitor = \
            map(np.int32, lines[lix].split(', '))

        # Read input/output parameters
        lix = 7
        g.fin_path = lines[lix].replace('\'', '').strip()
        lix = 8
        g.fout_path = lines[lix].replace('\'', '').strip()

        print('Input data file: {:s}'.format(g.fin_path))
        print('Output data file: {:s}'.format(g.fout_path))
        sys.stdout.flush()

        # Close parameter input file
        fil.close()

    # Broadcast user input (MPI)
    # intbuf = np.zeros((10), dtype=np.int32)
    floatbuf = np.zeros((10), dtype=np.float64)
    # fluid properties
    floatbuf[0:6] = [g.mu, g.gamma, g.Pr, g.R_g, g.k, g.D]
    g.comm.Bcast([floatbuf, g.MPI.DOUBLE], root=0)
    g.mu, g.gamma, g.Pr, g.R_g, g.k, g.D = floatbuf[0:6]
    # domain specification
    floatbuf[0:3] = [g.Lx, g.Ly, g.Lz]
    g.comm.Bcast([floatbuf, g.MPI.DOUBLE], root=0)
    g.Lx, g.Ly, g.Lz = floatbuf[0:3]
    # inlet conditions
    floatbuf[0:8] = [g.jet_height_y, g.jet_height_z, g.U_jet, g.V_jet,
                     g.W_jet, g.P_jet, g.T_jet, g.Phi_jet]
    g.comm.Bcast([floatbuf, g.MPI.DOUBLE], root=0)
    g.jet_height_y, g.jet_height_z, g.U_jet, g.V_jet, g.W_jet, g.P_jet, \
        g.T_jet, g.Phi_jet = floatbuf[0:8]
    # ambient conditions
    floatbuf[0:6] = [g.U_inf, g.V_inf, g.W_inf, g.P_inf, g.T_inf,
                     g.Phi_inf]
    g.comm.Bcast([floatbuf, g.MPI.DOUBLE], root=0)
    g.U_inf, g.V_inf, g.W_inf, g.P_inf, g.T_inf, g.Phi_inf = \
        floatbuf[0:6]
    # grid parameters
    g.nx = g.comm.bcast(g.nx, root=0)
    g.ny = g.comm.bcast(g.ny, root=0)
    g.nz = g.comm.bcast(g.nz, root=0)
    # timestep parameters
    floatbuf[0] = g.CFL_ref
    g.comm.Bcast([floatbuf, g.MPI.DOUBLE], root=0)
    g.CFL_ref = floatbuf[0]
    g.nsteps = g.comm.bcast(g.nsteps, root=0)
    g.nsave = g.comm.bcast(g.nsave, root=0)
    g.nmonitor = g.comm.bcast(g.nmonitor, root=0)
    # input/output parameters
    g.fin_path = g.comm.bcast(g.fin_path, root=0)
    g.fout_path = g.comm.bcast(g.fout_path, root=0)


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
        g.U[:, :, :] = 0.0
        g.V[:, :, :] = 0.0
        g.W[:, :, :] = 0.0
        g.P[:, :, :] = g.P_inf
        g.Phi[:, :, :] = 0.0
        _rho, _rho_u, _rho_v, _rho_w, _e_tot, _rho_phi = \
            eq.PrimToCons(g.Rho, g.U, g.V, g.W, g.P, g.Phi, g.gamma)
        g.Q[:, :, :, 0] = _rho
        g.Q[:, :, :, 1] = _rho_u
        g.Q[:, :, :, 2] = _rho_v
        g.Q[:, :, :, 3] = _rho_w
        g.Q[:, :, :, 4] = _e_tot
        g.Q[:, :, :, 5] = _rho_phi
    
        for i in range(g.nx):
            for j in range(g.ny):
                for k in range(g.nz):
                    if abs(g.yg[0,j,0]) < g.jet_height_y/2.0 and \
                            abs(g.zg[0,0,k]) < g.jet_height_z/2.0 and \
                            abs(g.xg[i,0,0]) < g.jet_height_y:
                        _u = g.U_jet * (1.0 - (2.0*g.yg[0,j,0]/g.jet_height_y)**2.0) * (1.0 - (g.xg[i,0,0]/g.jet_height_y))
                        g.Q[i,j,k,1] = g.Q[i,j,k,0] * _u



def read_input_data():
    """Read flow variables from data file fin_path

    filename format: 'fin_path.{tstep}.{rank}'
    Assumes exactly "nprocs" files exist
    """
    # Specify the input file(s)
    fin_path = g.fin_path

    # Read from the input file(s)
    fil = open(fin_path, 'rb')
    save_vars = pickle.load(fil)
    fil.close()

    # Set the flow variables
    g.xg_global = save_vars[0]
    g.yg_global = save_vars[1]
    g.zg_global = save_vars[2]
    Q_global = save_vars[3]

    i0 = g.i0_global[g.myrank]
    i1 = g.i1_global[g.myrank] + 1
    g.xg = g.xg_global[i0:i1, 0, 0]
    g.yg = g.yg_global[0, :, 0]
    g.zg = g.zg_global[0, 0, :]
    g.Q = Q_global[i0:i1, :, :, :]


def output_data():
    """Write flow state to output file fout_path

    filename format: 'fout_path.{tstep}.{rank}'
    """
    # Specify the output file
    fout_path = g.fout_path + '.{0:06d}'.format(g.tstep)

    if g.myrank == 0:
        print('Writing to {0:s}'.format(fout_path))
        sys.stdout.flush()

    # Collect the data to output
    full_q = np.zeros((g.nx_global + 1,
                       g.ny_global + 1,
                       g.nz_global + 1,
                       g.NVARS),
                      dtype=np.float64)
    # Send from rank > 0
    if g.myrank != 0:
        i0 = 1  # exclude first plane
        i1 = g.nx  # exclude last plane
        if g.myrank == g.nprocs-1:
            i1 = i1 + 1
        g.comm.Send([g.Q[i0:i1, :, :, :], g.MPI.DOUBLE],
                    dest=0,
                    tag=g.myrank)
        return  # rank > 0 processes done

    # Receive on rank == 0
    i0 = g.i0_global[0]
    i1 = g.i1_global[0]
    full_q[i0:i1, :, :, :] = g.Q[:-1, :, :, :]
    for src in range(1, g.nprocs):
        i0 = g.i0_global[src] + 1  # exclude first plane
        i1 = g.i1_global[src]  # exclude last plane
        if src == g.nprocs-1:
            i1 = i1 + 1
        g.comm.Recv([full_q[i0:i1, :, :, :], g.MPI.DOUBLE],
                    source=src,
                    tag=src)

    # Variables to save
    save_vars = [g.xg_global, g.yg_global, g.zg_global, full_q]

    # Write binary output
    fil = open(fout_path, 'wb')
    pickle.dump(save_vars, fil)
    fil.close()

    # # Specify the output file
    # fout_path = g.fout_path + \
    #     '.{0:06d}.{1:03d}'.format(g.tstep, g.myrank)

    # # Variables to save
    # save_vars = [g.xg, g.yg, g.zg, g.Q]

    # # Write binary output
    # fil = open(fout_path, 'wb')
    # pickle.dump(save_vars, fil)
    # fil.close()


#

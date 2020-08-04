"""initialize.py

Description:
    Module for initializing arrays and flow variables.
    Initializes MPI processes.
"""
import numpy as np
from mpi4py import MPI

import boundary_conditions as bc
import common as g
import input_output as io


def initialize():
    """Initialize simulation, flow variables, data and arrays"""

    # --- Initialize MPI -----------------------------------------------
    g.MPI = MPI
    g.comm = MPI.COMM_WORLD
    g.nprocs = g.comm.Get_size()
    g.myrank = g.comm.Get_rank()

    # --- Read User Input Parameters -----------------------------------
    io.read_input_parameters()

    # --- Parallel Grid Allocation -------------------------------------
    # split up the grid for parallel calculation
    # parallelize in streamwise direction
    if g.nprocs > g.nx:
        raise Exception("Too many processors for given nx")
    strides = np.linspace(0, g.nx, g.nprocs + 1)
    strides = np.floor(strides)
    g.i0_global = np.int32(strides[:g.nprocs])
    g.i1_global = np.int32(np.minimum(strides[1:] + 1, g.nx))
    strides = g.i1_global - g.i0_global
    # save global and local grid size
    g.nx_global = g.nx
    g.nx = strides[g.myrank]
    # normal and spanwise have no parallelization
    g.ny_global = g.ny
    g.nz_global = g.nz

    # --- Allocate Arrays ----------------------------------------------
    # Transport variable arrays
    g.Q = np.zeros((g.nx+1, g.ny+1, g.nz+1, g.NVARS))
    g.Qo = np.zeros((g.nx+1, g.ny+1, g.nz+1, g.NVARS))

    # Primitive variable arrays
    g.Rho = np.zeros((g.nx+1, g.ny+1, g.nz+1))
    g.U = np.zeros((g.nx+1, g.ny+1, g.nz+1))
    g.V = np.zeros((g.nx+1, g.ny+1, g.nz+1))
    g.W = np.zeros((g.nx+1, g.ny+1, g.nz+1))
    g.P = np.zeros((g.nx+1, g.ny+1, g.nz+1))
    g.Phi = np.zeros((g.nx+1, g.ny+1, g.nz+1))

    # RHS variables
    g.E = np.zeros((g.nx+1, g.ny+1, g.nz+1, g.NVARS))
    g.F = np.zeros((g.nx+1, g.ny+1, g.nz+1, g.NVARS))
    g.G = np.zeros((g.nx+1, g.ny+1, g.nz+1, g.NVARS))
    g.dEdx = np.zeros((g.nx+1, g.ny+1, g.nz+1, g.NVARS))
    g.dFdy = np.zeros((g.nx+1, g.ny+1, g.nz+1, g.NVARS))
    g.dGdz = np.zeros((g.nx+1, g.ny+1, g.nz+1, g.NVARS))

    # Grid Arrays
    g.xg_global = np.ndarray((g.nx_global+1, 1, 1))
    g.yg_global = np.ndarray((1, g.ny_global+1, 1))
    g.zg_global = np.ndarray((1, 1, g.nz_global+1))
    g.xg = np.ndarray((g.nx+1, 1, 1))
    g.yg = np.ndarray((1, g.ny+1, 1))
    g.zg = np.ndarray((1, 1, g.nz+1))

    # --- Time Variables -----------------------------------------------
    g.t = 0.0  # simulation time
    g.dt = 0.0  # timestep size
    g.tstep = 0  # timestep index

    # Runge Kutta Variables
    g.rk_step_1 = 'predictor'
    g.rk_step_2 = 'corrector'

    # --- Build Grids --------------------------------------------------
    # build the global grid arrays
    xg_temp = np.linspace(0, g.Lx, g.nx_global+1)
    if (g.ny_global % 2 == 1):
        yg_temp = np.linspace(-g.Ly/2, g.Ly/2, g.ny_global+1)
    else:
        raise Exception("Use odd values for ny")

    if (g.nz_global % 2 == 1):
        zg_temp = np.linspace(-g.Lz/2, g.Lz/2, g.nz_global+1)
    else:
        raise Exception('Use odd values for nz')

    # use shape to allow easy commuting with field variables
    g.xg_global[:, 0, 0] = xg_temp
    g.yg_global[0, :, 0] = yg_temp
    g.zg_global[0, 0, :] = zg_temp

    # assign the local grid arrays
    i0 = g.i0_global[g.myrank]
    i1 = g.i1_global[g.myrank] + 1
    g.xg[:, 0, 0] = g.xg_global[i0:i1, 0, 0]
    g.yg[0, :, 0] = g.yg_global[0, :, 0]
    g.zg[0, 0, :] = g.zg_global[0, 0, :]

    # --- Boundary Conditions ------------------------------------------
    # calculate boundary densities
    g.Rho_inf = g.P_inf / (g.R_g * g.T_inf)
    g.Rho_jet = g.P_jet / (g.R_g * g.T_jet)

    # --- Sponge -------------------------------------------------------
    # calculate the sponge damping factors
    x_sponge = g.x_sponge*g.Lx
    y_sponge = g.y_sponge*g.Ly/2
    z_sponge = g.z_sponge*g.Lz/2
    wall_dist = np.zeros((g.nx+1, g.ny+1, g.nz+1, 1))
    for i in range(g.nx+1):
        for j in range(g.ny+1):
            for k in range(g.nz+1):
                # calculate nondim dist from desired boundaries
                x1 = np.inf
                y0 = np.inf
                y1 = np.inf
                z0 = np.inf
                z1 = np.inf
                if x_sponge > 0:
                    x1 = abs(g.Lx - g.xg[i, 0, 0])/x_sponge
                if y_sponge > 0:
                    y0 = abs(g.yg[0, j, 0] + g.Ly/2)/y_sponge
                    y1 = abs(g.Ly/2 - g.yg[0, j, 0])/y_sponge
                if z_sponge > 0:
                    z0 = abs(g.zg[0, 0, k] + g.Lz/2)/z_sponge
                    z1 = abs(g.Lz/2 - g.zg[0, 0, k])/z_sponge
                # take minimum distance within sponge length
                wall_dist[i, j, k, 0] = np.min([x1, y0, y1, z0, z1, 1.0])
    g.sponge_fac = g.sponge_strength * (1.0 - wall_dist**g.a_sponge)

    # Calculate the reference condition
    g.Qref = np.zeros((1, 1, 1, g.NVARS))
    g.Qref[:, :, :, 0] = g.Rho_inf
    g.Qref[:, :, :, 1] = 0.0
    g.Qref[:, :, :, 2] = 0.0
    g.Qref[:, :, :, 3] = 0.0
    g.Qref[:, :, :, 4] = g.P_inf / (g.gamma - 1.0)
    g.Qref[:, :, :, 5] = 0.0

    # --- Initial Flow -------------------------------------------------
    io.init_flow()  # set the initial flow
    g.Qo[:, :, :, :] = g.Q[:, :, :, :]  # init prev flow state
    bc.apply_boundary_conditions()


#

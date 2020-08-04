"""boundary_conditions.py

Contains functions necessary to enforce boundary conditions of flow.

Includes physical boundary conditions at edges of domain as well as
numerical boundary conditions including updating interior ghost planes
between parallel tasks.
"""

import numpy as np

import common as g


def apply_boundary_conditions():
    """Apply boundary conditions to transported variables"""

    # inlet boundary
    if g.myrank == 0:
        # base inlet boundary
        apply_isothermal_wall('x0')
        # jet inlet condition
        for jj in range(g.ny):
            for kk in range(g.nz):
                if (abs(g.yg[0, jj, 0]) <= g.jet_height_y/2.0 and
                        abs(g.zg[0, 0, kk]) <= g.jet_height_z/2.0):
                    g.Q[0, jj, kk, 0] = g.Rho_jet
                    g.Q[0, jj, kk, 1] = g.Rho_jet * g.U_jet
                    g.Q[0, jj, kk, 2] = g.Rho_jet * g.V_jet
                    g.Q[0, jj, kk, 3] = g.Rho_jet * g.W_jet
                    g.Q[0, jj, kk, 4] = g.P_jet / (g.gamma-1) + \
                        0.5 * g.Rho_jet * g.U_jet**2
                    g.Q[0, jj, kk, 5] = g.Rho_jet * g.Phi_jet

    # outlet boundary
    if g.myrank == g.nprocs-1:
        apply_convective_bc('x1')

    communicate_internal_planes()

    # normal boundaries
    apply_pressure_bc('y0')
    apply_pressure_bc('y1')
    # apply_extrapolation_bc('y0')
    # apply_extrapolation_bc('y1')

    # spanwise boundaries
    apply_periodic_bc('z')
    # apply_extrapolation_bc('z0')
    # apply_extrapolation_bc('z1')
    # apply_pressure_bc('z0')
    # apply_pressure_bc('z1')


# --------------------------
# Communicate data between MPI planes
#
def communicate_internal_planes():

    # all processes with a chunk ahead of them
    if g.myrank < g.nprocs-2:
        g.comm.Isend(g.Q[-2, :, :, :], dest=g.myrank+1, tag=g.myrank)

    # zeroth process doesn't receive
    if g.myrank != 0:
        g.comm.Recv(g.Q[0, :, :, :], source=g.myrank-1, tag=g.myrank-1)

    # all processes with a chunk behind them
    if g.myrank > 0:
        g.comm.Isend(g.Q[1, :, :, :], dest=g.myrank-1, tag=g.myrank)

    # nprocs-1 process doesn't receive
    if g.myrank != g.nprocs-1:
        g.comm.Recv(g.Q[-1, :, :, :], source=g.myrank+1, tag=g.myrank+1)


# --------------------------
# Extrapolation BC
#  - phi_w = phi_w+1
#
def apply_extrapolation_bc(dirid):
    if dirid == 'x0':
        g.Q[0, :, :, :] = g.Q[1, :, :, :]
    elif dirid == 'x1':
        g.Q[g.nx, :, :, :] = g.Q[g.nx-1, :, :, :]
    elif dirid == 'y0':
        g.Q[:,    0, :, :] = g.Q[:,      1, :, :]
    elif dirid == 'y1':
        g.Q[:, g.ny, :, :] = g.Q[:, g.ny-1, :, :]
    elif dirid == 'z0':
        g.Q[:, :,    0, :] = g.Q[:, :,      1, :]
    elif dirid == 'z1':
        g.Q[:, :, g.nz, :] = g.Q[:, :, g.nz-1, :]
    else:
        msg = "Unknown dirid: {:s}".format(dirid)
        raise Exception(msg)


# --------------------------
# Convective outlet BC
#  - dU_i/dt = U_i * dU_i/dx_i
#  - U_i(n+1) = U_i + dt * (U_i * dU_i/dx_i)
#
def apply_convective_bc(dirid):
    if dirid == 'x1':
        U = g.Qo[g.nx, :, :, 1] / g.Qo[g.nx, :, :, 0]
        factor = g.dt / (g.xg[g.nx, 0, 0] - g.xg[g.nx-1, 0, 0]) * U[:, :]
        factor = factor.reshape(*np.shape(factor), 1)
        g.Q[g.nx, :, :, :] = g.Qo[g.nx, :, :, :] - \
            factor * (g.Qo[g.nx, :, :, :] - g.Qo[g.nx-1, :, :, :])
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


# --------------------------
# Pressure BC
#  - Rho = Rho_inf
#  - P = P_inf
#
def apply_pressure_bc(dirid):
    if dirid == 'y0':
        U = g.Q[:, 0, :, 1]/g.Q[:, 0, :, 0]
        V = g.Q[:, 0, :, 2]/g.Q[:, 0, :, 0]
        W = g.Q[:, 0, :, 3]/g.Q[:, 0, :, 0]
        g.Q[:, 0, :, 0] = g.Rho_inf
        g.Q[:, 0, :, 1] = g.Rho_inf * U
        g.Q[:, 0, :, 2] = g.Rho_inf * V
        g.Q[:, 0, :, 3] = g.Rho_inf * W
        RhoU2 = 0.5 * g.Rho_inf * (U**2 + V**2 + W**2)
        g.Q[:, 0, :, 4] = g.P_inf / (g.gamma - 1.0) + RhoU2
    elif dirid == 'y1':
        U = g.Q[:, g.ny, :, 1]/g.Q[:, g.ny, :, 0]
        V = g.Q[:, g.ny, :, 2]/g.Q[:, g.ny, :, 0]
        W = g.Q[:, g.ny, :, 3]/g.Q[:, g.ny, :, 0]
        g.Q[:, g.ny, :, 0] = g.Rho_inf
        g.Q[:, g.ny, :, 1] = g.Rho_inf * U
        g.Q[:, g.ny, :, 2] = g.Rho_inf * V
        g.Q[:, g.ny, :, 3] = g.Rho_inf * W
        RhoU2 = 0.5 * g.Rho_inf * (U**2 + V**2 + W**2)
        g.Q[:, g.ny, :, 4] = g.P_inf / (g.gamma - 1.0) + RhoU2
    elif dirid == 'z0':
        U = g.Q[:, :, 0, 1]/g.Q[:, :, 0, 0]
        V = g.Q[:, :, 0, 2]/g.Q[:, :, 0, 0]
        W = g.Q[:, :, 0, 3]/g.Q[:, :, 0, 0]
        g.Q[:, :, 0, 0] = g.Rho_inf
        g.Q[:, :, 0, 1] = g.Rho_inf * U
        g.Q[:, :, 0, 2] = g.Rho_inf * V
        g.Q[:, :, 0, 3] = g.Rho_inf * W
        RhoU2 = 0.5 * g.Rho_inf * (U**2 + V**2 + W**2)
        g.Q[:, :, 0, 4] = g.P_inf / (g.gamma - 1.0) + RhoU2
    elif dirid == 'z1':
        U = g.Q[:, :, g.nz, 1]/g.Q[:, :, g.nz, 0]
        V = g.Q[:, :, g.nz, 2]/g.Q[:, :, g.nz, 0]
        W = g.Q[:, :, g.nz, 3]/g.Q[:, :, g.nz, 0]
        g.Q[:, :, g.nz, 0] = g.Rho_inf
        g.Q[:, :, g.nz, 1] = g.Rho_inf * U
        g.Q[:, :, g.nz, 2] = g.Rho_inf * V
        g.Q[:, :, g.nz, 3] = g.Rho_inf * W
        RhoU2 = 0.5 * g.Rho_inf * (U**2 + V**2 + W**2)
        g.Q[:, :, g.nz, 4] = g.P_inf / (g.gamma - 1.0) + RhoU2
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


# ---------------------------
#  Isothermal wall BC
#   - U,V,W = 0 (no-slip, no-penetration)
#   - T = const.
#
def apply_isothermal_wall(dirid):
    if dirid == 'x0':
        g.Q[0, :, :, 1] = 0
        g.Q[0, :, :, 2] = 0
        g.Q[0, :, :, 3] = 0
        g.Q[0, :, :, 4] = g.Q[0, :, :, 0] * g.R_g / (g.gamma - 1.0) * g.T_inf
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


# ---------------------------
#  Periodic BC
#   - phi(n) = phi(1)
#   - phi(0) = phi(n-1)
#
def apply_periodic_bc(dirid):
    if dirid == 'z':
        g.Q[:, :, g.nz, :] = g.Q[:, :,      1, :]
        g.Q[:, :,    0, :] = g.Q[:, :, g.nz-1, :]
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


#

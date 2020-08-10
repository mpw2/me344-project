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

    # update internal ghost planes
    communicate_internal_planes(g.Q)

    # inlet boundary
    if g.myrank == 0:
        # base inlet boundary
        apply_isothermal_wall('x0')
        apply_pressure_bc('x0')
        # jet inlet condition
        for j in range(g.ny):
            for k in range(g.nz):
                if (abs(g.yg[0, j, 0]) <= g.jet_height_y/2.0 and
                        abs(g.zg[0, 0, k]) <= g.jet_height_z/2.0):
                    g.Q[0, j, k, 0] = g.Rho_jet
                    g.Q[0, j, k, 1] = g.Rho_jet * g.U_jet * \
                        (1.0 - (2.0 * g.yg[0, j, 0] / g.jet_height_y)**2.0) + \
                        0.01 * g.U_jet * (2.0 * np.random.rand() - 1.0)
                    g.Q[0, j, k, 2] = g.Rho_jet * g.V_jet + \
                        0.005 * g.U_jet * (2.0 * np.random.rand() - 1.0)
                    g.Q[0, j, k, 3] = g.Rho_jet * g.W_jet
                    g.Q[0, j, k, 4] = g.P_jet / (g.gamma-1.0) + \
                        0.5 * g.Rho_jet * g.U_jet**2.0
                    g.Q[0, j, k, 5] = g.Rho_jet * g.Phi_jet

    # outlet boundary
    if g.myrank == g.nprocs-1:
        apply_convective_bc('x1')

    # normal boundaries
    apply_convective_bc('y0')
    apply_convective_bc('y1')
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


def communicate_internal_planes(f_arr):
    """Communicate internal ghost planes between MPI processes"""

    sendbuf = np.zeros((1, g.ny+1, g.nz+1, g.NVARS), dtype=np.float64)
    recvbuf = np.zeros((1, g.ny+1, g.nz+1, g.NVARS), dtype=np.float64)

    # all processes with a chunk ahead of them
    if g.myrank < g.nprocs-1:
        sendbuf = f_arr[g.nx-1:g.nx, :, :, :]
        g.comm.Isend([sendbuf, g.MPI.DOUBLE], dest=g.myrank+1, tag=g.myrank)

    # zeroth process doesn't receive
    if g.myrank > 0:
        g.comm.Recv([recvbuf, g.MPI.DOUBLE], source=g.myrank-1, tag=g.myrank-1)
        f_arr[0:1, :, :, :] = recvbuf

    # all processes with a chunk behind them
    if g.myrank > 0:
        sendbuf = f_arr[1:2, :, :, :]
        g.comm.Isend([sendbuf, g.MPI.DOUBLE], dest=g.myrank-1, tag=g.myrank)

    # nprocs-1 process doesn't receive
    if g.myrank < g.nprocs-1:
        g.comm.Recv([recvbuf, g.MPI.DOUBLE], source=g.myrank+1, tag=g.myrank+1)
        f_arr[g.nx:g.nx+1, :, :, :] = recvbuf


def apply_extrapolation_bc(dirid):
    """Extrapolation BC
        - phi(0) = phi(1)
        - phi(n) = phi(n-1)
    """
    if dirid == 'x0':
        g.Q[0, :, :, :] = g.Q[1, :, :, :]
    elif dirid == 'x1':
        g.Q[g.nx, :, :, :] = g.Q[g.nx-1, :, :, :]
    elif dirid == 'y0':
        g.Q[:, 0, :, :] = g.Q[:, 1, :, :]
    elif dirid == 'y1':
        g.Q[:, g.ny, :, :] = g.Q[:, g.ny-1, :, :]
    elif dirid == 'z0':
        g.Q[:, :, 0, :] = g.Q[:, :, 1, :]
    elif dirid == 'z1':
        g.Q[:, :, g.nz, :] = g.Q[:, :, g.nz-1, :]
    else:
        msg = "Unknown dirid: {:s}".format(dirid)
        raise Exception(msg)


def apply_convective_bc(dirid):
    """Convective outlet BC
        - dU_i/dt = U_i * dU_i/dx_i
        - U_i(n+1) = U_i + dt * (U_i * dU_i/dx_i)
    """
    if dirid == 'x1':
        _u = g.Qo[g.nx, :, :, 1] / g.Qo[g.nx, :, :, 0]
        factor = g.dt / (g.xg[g.nx, 0, 0] - g.xg[g.nx-1, 0, 0]) * _u
        factor = factor.reshape(g.ny+1, g.nz+1, 1)
        g.Q[g.nx, :, :, :] = g.Qo[g.nx, :, :, :] - \
            factor * (g.Qo[g.nx, :, :, :] - g.Qo[g.nx-1, :, :, :])
    elif dirid == 'y0':
        _v = g.Qo[:, 0, :, 1] / g.Qo[:, 0, :, 0]
        factor = g.dt / (g.yg[0, 0, 0] - g.yg[0, 1, 0]) * _v
        factor = factor.reshape(g.nx+1, g.nz+1, 1)
        g.Q[:, 0, :, :] = g.Qo[:, 0, :, :] - \
            factor * (g.Qo[:, 0, :, :] - g.Qo[:, 1, :, :])
    elif dirid == 'y1':
        _v = g.Qo[:, g.ny, :, 1] / g.Qo[:, g.ny, :, 0]
        factor = g.dt / (g.yg[0, g.ny, 0] - g.yg[0, g.ny-1, 0]) * _v
        factor = factor.reshape(g.nx+1, g.nz+1, 1)
        g.Q[:, g.ny, :, :] = g.Qo[:, g.ny, :, :] - \
            factor * (g.Qo[:, g.ny, :, :] - g.Qo[:, g.ny-1, :, :])
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


def apply_pressure_bc(dirid):
    """Pressure BC
        - Rho = Rho_inf
        - P = P_inf
    """
    if dirid == 'y0':
        _u = g.Q[:, 0, :, 1]/g.Q[:, 0, :, 0]
        _v = g.Q[:, 0, :, 2]/g.Q[:, 0, :, 0]
        _w = g.Q[:, 0, :, 3]/g.Q[:, 0, :, 0]
        g.Q[:, 0, :, 0] = g.Rho_inf
        g.Q[:, 0, :, 1] = g.Rho_inf * _u
        g.Q[:, 0, :, 2] = g.Rho_inf * _v
        g.Q[:, 0, :, 3] = g.Rho_inf * _w
        rho_u2 = 0.5 * g.Rho_inf * (_u**2.0 + _v**2.0 + _w**2.0)
        g.Q[:, 0, :, 4] = g.P_inf / (g.gamma - 1.0) + rho_u2
    elif dirid == 'y1':
        _u = g.Q[:, g.ny, :, 1]/g.Q[:, g.ny, :, 0]
        _v = g.Q[:, g.ny, :, 2]/g.Q[:, g.ny, :, 0]
        _w = g.Q[:, g.ny, :, 3]/g.Q[:, g.ny, :, 0]
        g.Q[:, g.ny, :, 0] = g.Rho_inf
        g.Q[:, g.ny, :, 1] = g.Rho_inf * _u
        g.Q[:, g.ny, :, 2] = g.Rho_inf * _v
        g.Q[:, g.ny, :, 3] = g.Rho_inf * _w
        rho_u2 = 0.5 * g.Rho_inf * (_u**2.0 + _v**2.0 + _w**2.0)
        g.Q[:, g.ny, :, 4] = g.P_inf / (g.gamma - 1.0) + rho_u2
    elif dirid == 'z0':
        _u = g.Q[:, :, 0, 1]/g.Q[:, :, 0, 0]
        _v = g.Q[:, :, 0, 2]/g.Q[:, :, 0, 0]
        _w = g.Q[:, :, 0, 3]/g.Q[:, :, 0, 0]
        g.Q[:, :, 0, 0] = g.Rho_inf
        g.Q[:, :, 0, 1] = g.Rho_inf * _u
        g.Q[:, :, 0, 2] = g.Rho_inf * _v
        g.Q[:, :, 0, 3] = g.Rho_inf * _w
        rho_u2 = 0.5 * g.Rho_inf * (_u**2.0 + _v**2.0 + _w**2.0)
        g.Q[:, :, 0, 4] = g.P_inf / (g.gamma - 1.0) + rho_u2
    elif dirid == 'z1':
        _u = g.Q[:, :, g.nz, 1]/g.Q[:, :, g.nz, 0]
        _v = g.Q[:, :, g.nz, 2]/g.Q[:, :, g.nz, 0]
        _w = g.Q[:, :, g.nz, 3]/g.Q[:, :, g.nz, 0]
        g.Q[:, :, g.nz, 0] = g.Rho_inf
        g.Q[:, :, g.nz, 1] = g.Rho_inf * _u
        g.Q[:, :, g.nz, 2] = g.Rho_inf * _v
        g.Q[:, :, g.nz, 3] = g.Rho_inf * _w
        rho_u2 = 0.5 * g.Rho_inf * (_u**2.0 + _v**2.0 + _w**2.0)
        g.Q[:, :, g.nz, 4] = g.P_inf / (g.gamma - 1.0) + rho_u2
    elif dirid == 'x0':
        _u = g.Q[0, :, :, 1]/g.Q[0, :, :, 0]
        _v = g.Q[0, :, :, 2]/g.Q[0, :, :, 0]
        _w = g.Q[0, :, :, 3]/g.Q[0, :, :, 0]
        g.Q[0, :, :, 0] = g.Rho_inf
        g.Q[0, :, :, 1] = g.Rho_inf * _u
        g.Q[0, :, :, 2] = g.Rho_inf * _v
        g.Q[0, :, :, 3] = g.Rho_inf * _w
        rho_u2 = 0.5 * g.Rho_inf * (_u**2.0 + _v**2.0 + _w**2.0)
        g.Q[0, :, :, 4] = g.P_inf / (g.gamma - 1.0) + rho_u2
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


def apply_isothermal_wall(dirid):
    """Isothermal wall BC
        - U,V,W = 0 (no-slip, no-penetration)
        - T = const.
    """
    if dirid == 'x0':
        g.Q[0, :, :, 1] = 0.0
        g.Q[0, :, :, 2] = 0.0
        g.Q[0, :, :, 3] = 0.0
        g.Q[0, :, :, 4] = g.Q[0, :, :, 0] * g.R_g / (g.gamma - 1.0) * g.T_inf
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


def apply_periodic_bc(dirid):
    """Periodic BC
        - phi(n) = phi(1)
        - phi(0) = phi(n-1)
    """
    if dirid == 'z':
        g.Q[:, :, g.nz, :] = g.Q[:, :, 1, :]
        g.Q[:, :, 0, :] = g.Q[:, :, g.nz-1, :]
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


#

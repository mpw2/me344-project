import common as g
import numpy as np


# --------------------------------------------------
# Apply boundary conditions to transported variables
#
def apply_boundary_conditions():

    # # apply a rudimentary sponge
    # apply_sponge()

    # apply boundary conditions
    apply_isothermal_wall('x0')
    # apply_extrapolation_bc('x1')
    apply_convective_bc('x1')

    # apply_extrapolation_bc('y0')
    # apply_extrapolation_bc('y1')
    apply_pressure_bc('y0')
    apply_pressure_bc('y1')

    # apply_extrapolation_bc('z0')
    # apply_extrapolation_bc('z1')
    # apply_pressure_bc('z0')
    # apply_pressure_bc('z1')
    apply_periodic_bc('z')

    # apply inlet BC
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
#  - boundary normal velocity
#  - calculated from previous step
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
#  - set the total energy and density by
#    ambient pressure and temperature
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
#
def apply_periodic_bc(dirid):
    if dirid == 'z':
        g.Q[:, :, g.nz, :] = g.Q[:, :,      1, :]
        g.Q[:, :,    0, :] = g.Q[:, :, g.nz-1, :]
    else:
        msg = "BC not implemented for dirid: {:s}".format(dirid)
        raise Exception(msg)


# ---------------------------
# Apply a rudimentary sponge
#
def apply_sponge():

    # Apply the sponge to the field
    g.Q = (1.0 - g.sponge_fac) * g.Q + g.sponge_fac * g.Qref


#

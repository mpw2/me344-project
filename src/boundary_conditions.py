import common as g
import numpy as np

def apply_boundary_conditions():

    Rho_inf = g.P_inf / (g.R_g * g.T_inf)
    Rho_jet = g.P_jet / (g.R_g * g.T_jet)
    
    # apply left BC
    for jj in range(g.ny):
        if np.abs(g.yg[0,jj]) < g.jet_height:
            g.Q[0,jj,0] = Rho_jet
            g.Q[0,jj,1] = Rho_jet * g.U_jet
            g.Q[0,jj,2] = Rho_jet * g.V_jet
            g.Q[0,jj,3] = g.P_jet / (g.gamma-1) + 0.5*Rho_jet*g.U_jet**2
        else:
            g.Q[0,jj,0] = Rho_inf
            g.Q[0,jj,1] = 0
            g.Q[0,jj,2] = 0
            g.Q[0,jj,3] = g.P_inf / (g.gamma - 1)

    # apply top bc
    # extrapolation bc 
    g.Q[1:g.nx,g.ny,:] = 2 * g.Q[1:g.nx,g.ny-1,:] - g.Q[1:g.nx,g.ny-2,:]

    # apply bottom bc
    # extrapolation bc
    g.Q[1:g.nx,0,:] = 2 * g.Q[1:g.nx,1,:] - g.Q[1:g.nx,2,:]
 
    # apply right bc (do not include jmin or jmax)
    # extrapolation BC
    g.Q[g.nx,1:g.ny-1,:] = 2 * g.Q[g.nx-1,1:g.ny-1,:] - g.Q[g.nx-2,1:g.ny-1,:]
























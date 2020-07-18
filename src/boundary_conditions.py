import common as g
import numpy as np

def apply_boundary_conditions():

    apply_sponge()

    # APPLY LEFT BC
    for jj in range(g.ny):
        if abs( g.yg[0,jj] ) < g.jet_height/2.0:
            g.Q[0,jj,0] = g.Rho_jet
            g.Q[0,jj,1] = g.Rho_jet * g.U_jet
            g.Q[0,jj,2] = g.Rho_jet * g.V_jet
            g.Q[0,jj,3] = g.P_jet / (g.gamma-1) + 0.5*g.Rho_jet*g.U_jet**2
        else:
            # g.Q[0, jj, :] = g.Q[1, jj, :]
            g.Q[0,jj,0] = g.Rho_inf
            g.Q[0,jj,1] = 0
            g.Q[0,jj,2] = 0
            g.Q[0,jj,3] = g.P_inf / (g.gamma - 1)

    # APPLY RIGHT BC (do not include jmin or jmax)
    # extrapolation BC
    # g.Q[g.nx,1:g.ny-1,:] = 2 * g.Q[g.nx-1,1:g.ny-1,:] - g.Q[g.nx-2,1:g.ny-1,:]
    g.Q[g.nx, :, :] = g.Q[g.nx-1, :, :]
    
    # APPLY TOP BC
    # extrapolation bc 
    # g.Q[1:g.nx,g.ny,:] = 2 * g.Q[1:g.nx,g.ny-1,:] - g.Q[1:g.nx,g.ny-2,:]
    g.Q[:, g.ny, :] = g.Q[:, g.ny-1, :]
    # # slip bc 
    # g.Q[1:g.nx,g.ny,0] = g.Rho_inf
    # g.Q[1:g.nx,g.ny,1] = g.Q[1:g.nx,g.ny-1,1]
    # g.Q[1:g.nx,g.ny,2] = 0
    # g.Q[1:g.nx,g.ny,3] = g.P_inf/(g.gamma-1.0) + 0.5*g.Rho_inf*(g.Q[1:g.nx,g.ny,1]**2) 

    # APPLY BOTTOM BC
    # extrapolation bc
    #g.Q[1:g.nx,0,:] = 2 * g.Q[1:g.nx,1,:] - g.Q[1:g.nx,2,:]
    g.Q[:, 0, :] = g.Q[:, 1, :]
    # # slip bc
    # g.Q[1:g.nx,0,0] = g.Rho_inf
    # g.Q[1:g.nx,0,1] = g.Q[1:g.nx,1,1]
    # g.Q[1:g.nx,0,2] = 0
    # g.Q[1:g.nx,0,3] = g.P_inf/(g.gamma-1.0) + 0.5*g.Rho_inf*(g.Q[1:g.nx,0,1]**2) 
     

def apply_sponge():
    
    # Apply the sponge to the field
    g.Q = (1.0 - g.sponge_fac) * g.Q + g.sponge_fac * g.Qref


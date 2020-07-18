import common as g
import numpy as np
import input_output as io
import boundary_conditions as bc

def initialize():

    # Transport variable arrays
    g.Q = np.zeros((g.nx+1,g.ny+1,g.NVARS))
    
    # Primitive variable arrays
    g.Rho = np.zeros((g.nx+1,g.ny+1))
    g.U = np.zeros((g.nx+1,g.ny+1))
    g.V = np.zeros((g.nx+1,g.ny+1))
    g.P = np.zeros((g.nx+1,g.ny+1))

    # Time variables
    g.t = 0.0
    g.tstep = 0
    
    # Build the grid
    xg_temp = np.linspace(0,g.Lx,g.nx+1)
    if ( g.ny % 2 == 1 ):
        yg_temp = np.linspace(-g.Ly/2,g.Ly/2,g.ny+1)
    else:
        raise Exception("Use odd values for ny")
    # use shape to allow easy commuting with field variables
    g.xg = np.ndarray((g.nx+1,1))
    g.yg = np.ndarray((1,g.ny+1))
    g.xg[:,0] = xg_temp
    g.yg[0,:] = yg_temp
    

    g.Rho_inf = g.P_inf / (g.R_g * g.T_inf)
    g.Rho_jet = g.P_jet / (g.R_g * g.T_jet)

    # calculate the sponge damping factors
    x_sponge = g.x_sponge*g.Lx
    y_sponge = g.y_sponge*g.Ly/2
    wall_dist = np.zeros((g.nx+1,g.ny+1,1))
    for i in range(g.nx+1):
        for j in range(g.ny+1):
            # calculate nondim dist from desired boundaries
            x1 = abs(g.Lx - g.xg[i,0])/x_sponge
            y0 = abs(g.yg[0,j] + g.Ly/2)/y_sponge
            y1 = abs(g.Ly/2 - g.yg[0,j])/y_sponge
            # take minimum distance within sponge length
            wall_dist[i,j,0] = np.min([x1,y0,y1,1.0])
    g.sponge_fac = 1.0 - wall_dist**g.a_sponge
    
    # Calculate the reference condition
    g.Qref = np.zeros((1,1,g.NVARS))
    g.Qref[:,:,0] = g.Rho_inf
    g.Qref[:,:,1] = 0.0
    g.Qref[:,:,2] = 0.0
    g.Qref[:,:,3] = g.P_inf / (g.gamma - 1.0)
    
    # --------------------------
    # Initialize the flow field
    io.init_flow()
    bc.apply_boundary_conditions()


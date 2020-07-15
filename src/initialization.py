import common as g
import numpy as np

def initialize():

    # Transport variable arrays
    g.Q = np.zeros((g.nx+1,g.ny+1,g.NVARS))
    
    # Primitive variable arrays
    g.Rho = np.zeros((g.nx+1,g.ny+1))
    g.U = np.zeros((g.nx+1,g.ny+1))
    g.V = np.zeros((g.nx+1,g.ny+1))
    g.P = np.zeros((g.nx+1,g.ny+1))

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
    

















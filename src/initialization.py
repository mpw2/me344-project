from common import *

def initialize():
    # Output variables
    global Q
    global Rho, U, V, P

    # Transport variable arrays
    Q = np.zeros((nx,ny,nvars))
    
    # Primitive variable arrays
    Rho = np.zeros((nx,ny))
    U = np.zeros((nx,ny))
    V = np.zeros((nx,ny))
    P = np.zeros((nx,ny))

    # Build the grid
    xg_temp = np.linspace(0,Lx,nx)
    if ( ny % 2 == 1 ):
        yg_temp = np.linspace(-Ly/2,Ly/2,ny)
    else:
        raise Exception("Use odd values for ny")
    # use shape to allow easy commuting with field variables
    xg = np.ndarray((nx,1))
    yg = np.ndarray((1,ny))
    xg[:,0] = xg_temp
    yg[0,:] = yg_temp
    

















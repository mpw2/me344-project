import common as g
import numpy as np

# ---------------------------------------------------
# Compute Finite Difference in y direction
#
# Input:
#     phi    : quantity
#     x      : x-dir grid
#     y      : y-dir grid
#     dir_id : 0 - Forward, 1 - Backward, 2 - Central
# Output:
#     d(phi)/dy : same dimensions as phi
# ---------------------------------------------------
def compute_x_deriv(phi, x, y, dir_id):
    nx = x.shape[0]-1
    ny = y.shape[1]-1
    dphi = np.zeros((nx+1,ny+1))
    if dir_id == 0:
        dphi[    nx, :] = (phi[  nx, :] - phi[  nx-1, :]) / (x[  nx, :] - x[  nx-1, :])
        dphi[0:nx-1, :] = (phi[1:nx, :] - phi[0:nx-1, :]) / (x[1:nx, :] - x[0:nx-1, :])
    if dir_id == 1:
        dphi[     0, :] = (phi[   1, :] - phi[     0, :]) / (x[   1, :] - x[     0, :])
        dphi[  1:nx, :] = (phi[1:nx, :] - phi[0:nx-1, :]) / (x[1:nx, :] - x[0:nx-1, :]) 
    if dir_id == 2:
        dphi[     0, :] = (phi[   1, :] - phi[     0, :]) / (x[   1, :] - x[     0, :])
        dphi[    nx, :] = (phi[  nx, :] - phi[  nx-1, :]) / (x[  nx, :] - x[  nx-1, :])
        dphi[1:nx-1, :] = (phi[2:nx, :] - phi[0:nx-2, :]) / (x[2:nx, :] - x[0:nx-2, :]) 
    return dphi    

# ---------------------------------------------------
# Compute Finite Difference in y direction
#
# Input:
#     phi    : quantity
#     x      : x-dir grid
#     y      : y-dir grid
#     dir_id : 0 - Forward, 1 - Backward, 2 - Central
# Output:
#     d(phi)/dy : same dimensions as phi
# ---------------------------------------------------
def compute_y_deriv(phi, x, y, dir_id):
    nx = x.shape[0]-1
    ny = y.shape[1]-1
    dphi = np.zeros((nx+1,ny+1))
    if dir_id == 0:
        dphi[:,     ny] = (phi[:,   ny] - phi[:,   ny-1]) / (y[:,   ny] - y[:,   ny-1])
        dphi[:, 0:ny-1] = (phi[:, 1:ny] - phi[:, 0:ny-1]) / (y[:, 1:ny] - y[:, 0:ny-1]) 
    if dir_id == 1:
        dphi[:,      0] = (phi[:,    1] - phi[:,      0]) / (y[:,    1] - y[:,      0])
        dphi[:,   1:ny] = (phi[:, 1:ny] - phi[:, 0:ny-1]) / (y[:, 1:ny] - y[:, 0:ny-1]) 
    if dir_id == 2:
        dphi[:,      0] = (phi[:,    1] - phi[:,      0]) / (y[:,    1] - y[:,      0])
        dphi[:,     ny] = (phi[:,   ny] - phi[:,   ny-1]) / (y[:,   ny] - y[:,   ny-1])
        dphi[:, 1:ny-1] = (phi[:, 2:ny] - phi[:, 0:ny-2]) / (y[:, 2:ny] - y[:, 0:ny-2]) 
    return dphi    

def ConsToPrim(Q):
    Rho_ = np.squeeze(Q[:,:,0])
    U_ = np.squeeze(Q[:,:,1] / Q[:,:,0])
    V_ = np.squeeze(Q[:,:,2] / Q[:,:,0])
    P_ = np.squeeze((g.gamma - 1) * (Q[:,:,3] - 0.5 / Q[:,:,0] * (Q[:,:,1] + Q[:,:,2])**2))

    return Rho_, U_, V_, P_

def PrimToCons(Rho,U,V,P):
    rhoU_ = Rho * U
    rhoV_ = Rho * V
    Et_ = P / (g.gamma - 1) + 0.5 * Rho * (U**2 + V**2)

    return Rho, rhoU_, rhoV_, Et_

def Tauxx(U,x,y,mu,step):

    if step == 'predictor':
        tau_xx = 2 * mu * compute_x_deriv(U,x,y,1)

    elif step == 'corrector':
        tau_xx = 2 * mu * compute_x_deriv(U,x,y,0)

    else:
        raise Exception('Invalid Step')

    return tau_xx

def Tauyy(V,x,y,mu,step):

    if step == 'predictor':
        tau_yy = 2 * mu * compute_y_deriv(V,x,y,1)
    elif step == 'corrector':
        tau_yy = 2 * mu * compute_y_deriv(V,x,y,0)
    else:
        raise Exception('Invalid Step')

    return tau_yy

def Qx(T,x,y,k,step):

    if step == 'predictor':
        qx = -1 * k * compute_x_deriv(T,x,y,1)
    elif step == 'corrector':
        qx = -1 * k * compute_x_deriv(T,x,y,0)
    else:
        raise Exception('Invalid Step')

    return qx

def Qy(T,x,y,k,step):

    if step == 'predictor':
        qy = -1 * k * compute_y_deriv(T,x,y,1)
    elif step == 'corrector':
        qy = -1 * k * compute_y_deriv(T,x,y,0)
    else:
        raise Exception('Invalid Step')

    return qy

def Tauxy(U,V,x,y,mu,flux_dir,step):

    if step == 'predictor':
        if flux_dir == 0:
            tau_xy = mu * (compute_y_deriv(U,x,y,2) + compute_x_deriv(V,x,y,1))
        elif flux_dir == 1:
            tau_xy = mu * (compute_y_deriv(U,x,y,2) + compute_x_deriv(V,x,y,0))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 'corrector':

        if flux_dir == 0:
            tau_xy = mu * (compute_y_deriv(U,x,y,1) + compute_x_deriv(V,x,y,2))
        elif flux_dir == 1:
            tau_xy = mu * (compute_y_deriv(U,x,y,0) + compute_x_deriv(V,x,y,2))
        else:
            raise Exception('Invalid Flux Direction')

    else:
        raise Exception('Invalid Step')

    return tau_xy

def compE(Q,x,y,Rgas,mu,kappa,gamma,step):
    nx = x.shape[0]
    ny = y.shape[1]
    E = np.zeros([nx,ny,g.NVARS])

    Rho, U, V, P = ConsToPrim(Q)

    T = P / (Rho * Rgas)

    tau_xx = Tauxx(U,x,y,mu,step)
    tau_xy = Tauxy(U,V,x,y,mu,0,step)
    qx = Qx(T,x,y,kappa,step)

    E[:,:,0] = Rho * U
    E[:,:,1] = Rho * U**2 + P - tau_xx
    E[:,:,2] = Rho * U * V - tau_xy
    E[:,:,3] = (Q[:,:,-1] + P) * U - U * tau_xx - V * tau_xy + qx

    return E

def compF(Q,x,y,Rgas,mu,kappa,gamma,step):
    nx = x.shape[0]
    ny = y.shape[1]
    F = np.zeros([nx,ny,g.NVARS])

    Rho, U, V, P = ConsToPrim(Q)

    T = P / (Rho * Rgas)

    tau_yy = Tauyy(V,x,y,mu,step)
    tau_xy = Tauxy(U,V,x,y,mu,1,step)
    qy = Qy(T,x,y,kappa,step)

    F[:,:,0] = Rho * V
    F[:,:,1] = Rho * U * V - tau_xy
    F[:,:,2] = Rho * V**2 + P - tau_yy
    F[:,:,3] = (Q[:,:,-1] + P) * V - U * tau_xy - V * tau_yy + qy

    return F

def compRHS(Q,x,y,step):
    nx = x.shape[0]
    ny = y.shape[1]

    dEdx = np.zeros((nx,ny,g.NVARS))
    dFdy = np.zeros((nx,ny,g.NVARS))

    E = compE(Q, x, y, g.R_g, g.mu, g.k, g.gamma, step)
    F = compF(Q, x, y, g.R_g, g.mu, g.k, g.gamma, step)
    
    dir_id = None
    
    # format of Q[x,y,0-3] 
    if step == 'predictor':
        dir_id = 0
    elif step == 'corrector':
        dir_id = 1
    else:
        raise Exception('Invalid derivative dir_id')

    for ii in range(g.NVARS):
        dEdx[:,:,ii] = compute_x_deriv(E[:,:,ii],x,y,dir_id)
        dFdy[:,:,ii] = compute_y_deriv(F[:,:,ii],x,y,dir_id)

    rhs = -1 * (dEdx + dFdy)
    
    return rhs




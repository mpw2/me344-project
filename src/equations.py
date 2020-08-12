"""equations.py

Description:
 - Contains the governing equations of the flow and forcing terms.
 - Contains the numerical operators for computing derivatives.
 - Contains functions to convert conserved and primitive variables.
"""

# import numba as nb
import numpy as np
import numba as nb

import common as g
import boundary_conditions as bc


# @nb.jit(nopython=True)
def compute_x_deriv(phi, x, y, z, dir_id):
    """Compute Finite Difference in x direction
    Input:
        phi    : quantity
        x      : x-dir grid
        y      : y-dir grid
        z      : z-dir grid
        dir_id : 0 - Forward, 1 - Backward, 2 - Central
    Output:
        d(phi)/dy : same dimensions as phi
    """
    nx = x.shape[0]-1
    ny = y.shape[1]-1
    nz = z.shape[2]-1
    dphi = np.zeros((nx+1, ny+1, nz+1), dtype=np.float64)
    if dir_id == 0:
        dphi[nx, :, :] = (phi[nx, :, :] - phi[nx-1, :, :]) / \
                         (x[nx, :, :] - x[nx-1, :, :])
        dphi[0:nx, :, :] = (phi[1:nx+1, :, :] - phi[0:nx, :, :]) / \
                           (x[1:nx+1, :, :] - x[0:nx, :, :])
    if dir_id == 1:
        dphi[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / \
                        (x[1, :, :] - x[0, :, :])
        dphi[1:nx+1, :, :] = (phi[1:nx+1, :, :] - phi[0:nx, :, :]) / \
                             (x[1:nx+1, :, :] - x[0:nx, :, :])
    if dir_id == 2:
        dphi[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / \
                        (x[1, :, :] - x[0, :, :])
        dphi[nx, :, :] = (phi[nx, :, :] - phi[nx-1, :, :]) / \
                         (x[nx, :, :] - x[nx-1, :, :])
        dphi[1:nx, :, :] = (phi[2:nx+1, :, :] - phi[0:nx-1, :, :]) / \
                           (x[2:nx+1, :, :] - x[0:nx-1, :, :])
    return dphi


# @nb.jit(nopython=True)
def compute_y_deriv(phi, x, y, z, dir_id):
    """Compute Finite Difference in y direction
    Input:
        phi    : quantity
        x      : x-dir grid
        y      : y-dir grid
        z      : z-dir grid
        dir_id : 0 - Forward, 1 - Backward, 2 - Central
    Output:
        d(phi)/dy : same dimensions as phi
    """
    nx = x.shape[0]-1
    ny = y.shape[1]-1
    nz = z.shape[2]-1
    dphi = np.zeros((nx+1, ny+1, nz+1), dtype=np.float64)
    if dir_id == 0:
        dphi[:, ny, :] = (phi[:, ny, :] - phi[:, ny-1, :]) / \
                         (y[:, ny, :] - y[:, ny-1, :])
        dphi[:, 0:ny, :] = (phi[:, 1:ny+1, :] - phi[:, 0:ny, :]) / \
                           (y[:, 1:ny+1, :] - y[:, 0:ny, :])
    if dir_id == 1:
        dphi[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / \
                        (y[:, 1, :] - y[:, 0, :])
        dphi[:, 1:ny+1, :] = (phi[:, 1:ny+1, :] - phi[:, 0:ny, :]) / \
                             (y[:, 1:ny+1, :] - y[:, 0:ny, :])
    if dir_id == 2:
        dphi[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / \
                        (y[:, 1, :] - y[:, 0, :])
        dphi[:, ny, :] = (phi[:, ny, :] - phi[:, ny-1, :]) / \
                         (y[:, ny, :] - y[:, ny-1, :])
        dphi[:, 1:ny, :] = (phi[:, 2:ny+1, :] - phi[:, 0:ny-1, :]) / \
                           (y[:, 2:ny+1, :] - y[:, 0:ny-1, :])
    return dphi


# @nb.jit(nopython=True)
def compute_z_deriv(phi, x, y, z, dir_id):
    """Compute Finite Difference in z direction
    Input:
        phi    : quantity
        x      : x-dir grid
        y      : y-dir grid
        z      : z-dir grid
        dir_id : 0 - Forward, 1 - Backward, 2 - Central
    Output:
        d(phi)/dz : same dimensions as phi
    """
    nx = x.shape[0]-1
    ny = y.shape[1]-1
    nz = z.shape[2]-1
    dphi = np.zeros((nx+1, ny+1, nz+1), dtype=np.float64)
    if dir_id == 0:
        dphi[:, :, nz] = (phi[:, :, nz] - phi[:, :, nz-1]) / \
                         (z[:, :, nz] - z[:, :, nz-1])
        dphi[:, :, 0:nz] = (phi[:, :, 1:nz+1] - phi[:, :, 0:nz]) / \
                           (z[:, :, 1:nz+1] - z[:, :, 0:nz])
    if dir_id == 1:
        dphi[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / \
                        (z[:, :, 1] - z[:, :, 0])
        dphi[:, :, 1:nz+1] = (phi[:, :, 1:nz+1] - phi[:, :, 0:nz]) / \
                             (z[:, :, 1:nz+1] - z[:, :, 0:nz])
    if dir_id == 2:
        dphi[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / \
                        (z[:, :, 1] - z[:, :, 0])
        dphi[:, :, nz] = (phi[:, :, nz] - phi[:, :, nz-1]) / \
                         (z[:, :, nz] - z[:, :, nz-1])
        dphi[:, :, 1:nz] = (phi[:, :, 2:nz+1] - phi[:, :, 0:nz-1]) / \
                           (z[:, :, 2:nz+1] - z[:, :, 0:nz-1])
    return dphi


# @nb.jit(nopython=True)
def ConsToPrim(Q, gamma):
    """ConsToPrim - Conserved variables to primitive variables
    Input:
     - Q     : Vector of conserved variables (nx, ny, nz, nvars)
     - gamma : ratio of cp/cv
    Output:
     - Rho     : Density
     - U, V, W : Velocity components
     - P       : Pressure
     - Phi     : Passive Scalar
    """
    _rho = Q[:, :, :, 0]
    _u = Q[:, :, :, 1] / Q[:, :, :, 0]
    _v = Q[:, :, :, 2] / Q[:, :, :, 0]
    _w = Q[:, :, :, 3] / Q[:, :, :, 0]
    _p = (gamma - 1) * (Q[:, :, :, 4] - 0.5 / Q[:, :, :, 0] *
                        (Q[:, :, :, 1] + Q[:, :, :, 2] + Q[:, :, :, 3])**2.0)
    _phi = Q[:, :, :, 5] / Q[:, :, :, 0]

    return _rho, _u, _v, _w, _p, _phi


# @nb.jit(nopython=True)
def PrimToCons(rho, u, v, w, p, phi, gamma):
    """PrimToCons - Primitive variables to conserved variables
    Input:
     - Rho     : Density
     - U, V, W : Velocity components
     - P       : Pressure
     - Phi     : Passive scalar
    Output:
     - Rho       : Density
     - Rho U,V,W : Momentum
     - Et        : Total energy
     - Rho Phi   : Passive scalar
    """
    _rho_u = rho * u
    _rho_v = rho * v
    _rho_w = rho * w
    _e_tot = p / (gamma - 1.0) + 0.5 * rho * (u**2.0 + v**2.0 + w**2.0)
    _rho_phi = rho * phi

    return rho, _rho_u, _rho_v, _rho_w, _e_tot, _rho_phi


# @nb.jit(nopython=True)
def Tauxx(U, x, y, z, mu, step):
    """Calculate stress tensor component xx"""
    if step == 0:
        tau_xx = 2.0 * mu * compute_x_deriv(U, x, y, z, 1)
    elif step == 1:
        tau_xx = 2.0 * mu * compute_x_deriv(U, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_xx


# @nb.jit(nopython=True)
def Tauyy(V, x, y, z, mu, step):
    """Calculate stress tensor component yy"""
    if step == 0:
        tau_yy = 2.0 * mu * compute_y_deriv(V, x, y, z, 1)
    elif step == 1:
        tau_yy = 2.0 * mu * compute_y_deriv(V, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_yy


# @nb.jit(nopython=True)
def Tauzz(W, x, y, z, mu, step):
    """Calculate stress tensor component zz"""
    if step == 0:
        tau_zz = 2.0 * mu * compute_z_deriv(W, x, y, z, 1)
    elif step == 1:
        tau_zz = 2.0 * mu * compute_z_deriv(W, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_zz


# @nb.jit(nopython=True)
def Qx(T, x, y, z, k, step):
    """Calculate temperature diffusion in x"""
    if step == 0:
        qx = -1.0 * k * compute_x_deriv(T, x, y, z, 1)
    elif step == 1:
        qx = -1.0 * k * compute_x_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qx


# @nb.jit(nopython=True)
def Qy(T, x, y, z, k, step):
    """Calculate temperature diffusion in y"""
    if step == 0:
        qy = -1.0 * k * compute_y_deriv(T, x, y, z, 1)
    elif step == 1:
        qy = -1.0 * k * compute_y_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qy


# @nb.jit(nopython=True)
def Qz(T, x, y, z, k, step):
    """Calculate temperature diffusion in z"""
    if step == 0:
        qz = -1.0 * k * compute_z_deriv(T, x, y, z, 1)
    elif step == 1:
        qz = -1.0 * k * compute_z_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qz


# @nb.jit(nopython=True)
def Phix(Phi, x, y, z, D, step):
    """Calculate scalar diffusion in x"""
    if step == 0:
        phix = D * compute_x_deriv(Phi, x, y, z, 1)
    elif step == 1:
        phix = D * compute_x_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phix


# @nb.jit(nopython=True)
def Phiy(Phi, x, y, z, D, step):
    """Calculate scalar diffusion in y"""
    if step == 0:
        phiy = D * compute_y_deriv(Phi, x, y, z, 1)
    elif step == 1:
        phiy = D * compute_y_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phiy


# @nb.jit(nopython=True)
def Phiz(Phi, x, y, z, D, step):
    """Calculate scalar diffusion in z"""
    if step == 0:
        phiz = D * compute_z_deriv(Phi, x, y, z, 1)
    elif step == 1:
        phiz = D * compute_z_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phiz


# @nb.jit(nopython=True)
def Tauxy(U, V, x, y, z, mu, flux_dir, step):
    """Calculate stress tensor component xy"""
    if step == 0:
        if flux_dir == 0:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 2) +
                           compute_x_deriv(V, x, y, z, 1))
        elif flux_dir == 1:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 1) +
                           compute_x_deriv(V, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 1:
        if flux_dir == 0:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 2) +
                           compute_x_deriv(V, x, y, z, 0))
        elif flux_dir == 1:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 0) +
                           compute_x_deriv(V, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    else:
        raise Exception('Invalid Step')

    return tau_xy


# @nb.jit(nopython=True)
def Tauxz(U, W, x, y, z, mu, flux_dir, step):
    """Calculate stress tensor component xz"""
    if step == 0:
        if flux_dir == 0:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 2) +
                           compute_x_deriv(W, x, y, z, 1))
        elif flux_dir == 2:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 1) +
                           compute_x_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 1:
        if flux_dir == 0:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 2) +
                           compute_x_deriv(W, x, y, z, 0))
        elif flux_dir == 2:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 0) +
                           compute_x_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    return tau_xz


# @nb.jit(nopython=True)
def Tauyz(V, W, x, y, z, mu, flux_dir, step):
    """Calculate stress tensor component yz"""
    if step == 0:
        if flux_dir == 1:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 2) +
                           compute_y_deriv(W, x, y, z, 1))
        elif flux_dir == 2:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 1) +
                           compute_y_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 1:
        if flux_dir == 1:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 2) +
                           compute_y_deriv(W, x, y, z, 0))
        elif flux_dir == 2:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 0) +
                           compute_y_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    return tau_yz


# @nb.jit(nopython=True)
def comp_mu_sgs(Q, x, y, z):
    c_vre = 2.5 * (0.1)**2.0
    
    nx = x.shape[0]
    ny = y.shape[1]
    nz = z.shape[2]
    dx = np.zeros((nx, 1, 1), dtype=np.float64)
    dy = np.zeros((1, ny, 1), dtype=np.float64)
    dz = np.zeros((1, 1, nz), dtype=np.float64)
    dx[1:, 0, 0] = x[1:, 0, 0] - x[:nx-1, 0, 0]
    dy[0, 1:, 0] = y[0, 1:, 0] - y[0, :ny-1, 0]
    dz[0, 0, 1:] = z[0, 0, 1:] - z[0, 0, :nz-1]
    dx[0, 0, 0] = dx[1, 0, 0]
    dy[0, 0, 0] = dy[0, 1, 0]
    dz[0, 0, 0] = dz[0, 0, 1]
    
    rho = Q[:, :, :, 0]
    U = Q[:, :, :, 1] / rho
    V = Q[:, :, :, 2] / rho
    W = Q[:, :, :, 3] / rho
    beta = np.zeros((nx, ny, nz, 6), dtype=np.float64)
    alpha = np.zeros((nx, ny, nz, 9), dtype=np.float64)

    #        | 0 3 4 |
    # A_ij = | 6 1 5 |
    #        | 7 8 2 |
    alpha[:, :, :, 0] = compute_x_deriv(U, x, y, z, 2)
    alpha[:, :, :, 1] = compute_y_deriv(V, x, y, z, 2)
    alpha[:, :, :, 2] = compute_z_deriv(W, x, y, z, 2)
    alpha[:, :, :, 3] = compute_x_deriv(V, x, y, z, 2)
    alpha[:, :, :, 4] = compute_x_deriv(W, x, y, z, 2)
    alpha[:, :, :, 5] = compute_y_deriv(W, x, y, z, 2)
    alpha[:, :, :, 6] = compute_y_deriv(U, x, y, z, 2)
    alpha[:, :, :, 7] = compute_z_deriv(U, x, y, z, 2)
    alpha[:, :, :, 8] = compute_z_deriv(V, x, y, z, 2)
    
    beta[:, :, :, 0] = dx**2.0*alpha[:, :, :, 0]**2.0 + \
        dy**2.0*alpha[:, :, :, 6]**2.0 + \
        dz**2.0*alpha[:, :, :, 7]**2.0
    beta[:, :, :, 1] = dx**2.0*alpha[:, :, :, 3]**2.0 + \
        dy**2.0*alpha[:, :, :, 1]**2.0 + \
        dz**2.0*alpha[:, :, :, 8]**2.0
    beta[:, :, :, 2] = dx**2.0*alpha[:, :, :, 4]**2.0 + \
        dy**2.0*alpha[:, :, :, 5]**2.0 + \
        dz**2.0*alpha[:, :, :, 2]**2.0
    beta[:, :, :, 3] = dx**2.0*alpha[:, :, :, 0]*alpha[:, :, :, 3] + \
        dy**2.0*alpha[:, :, :, 6]*alpha[:, :, :, 1] + \
        dz**2.0*alpha[:, :, :, 7]*alpha[:, :, :, 8]
    beta[:, :, :, 4] = dx**2.0*alpha[:, :, :, 0]*alpha[:, :, :, 4] + \
        dy**2.0*alpha[:, :, :, 6]*alpha[:, :, :, 5] + \
        dz**2.0*alpha[:, :, :, 7]*alpha[:, :, :, 2]
    beta[:, :, :, 5] = dx**2.0*alpha[:, :, :, 3]*alpha[:, :, :, 4] + \
        dy**2.0*alpha[:, :, :, 1]*alpha[:, :, :, 5] + \
        dz**2.0*alpha[:, :, :, 8]*alpha[:, :, :, 2]

    B_beta = beta[:, :, :, 0]*beta[:, :, :, 1] - beta[:, :, :, 3]**2.0 + \
        beta[:, :, :, 0]*beta[:, :, :, 2] - beta[:, :, :, 4]**2.0 + \
        beta[:, :, :, 1]*beta[:, :, :, 2] - beta[:, :, :, 5]**2.0

    a_ija_ij = alpha[:,:,:,0]**2.0 + alpha[:,:,:,1]**2.0 + \
        alpha[:,:,:,2]**2.0 + alpha[:,:,:,3]**2.0 + alpha[:,:,:,4]**2.0 + \
        alpha[:,:,:,5]**2.0 + alpha[:,:,:,6]**2.0 + alpha[:,:,:,7]**2.0 + \
        alpha[:,:,:,8]**2.0
   
    mask = np.abs(a_ija_ij) > 1e-8    
    nu_e = c_vre * np.sqrt(np.divide(B_beta, a_ija_ij, where=mask))
    mu_e = rho * nu_e
 
    mu_e[0,:,:] = 0.0
    mu_e[-1,:,:] = 0.0
    mu_e[:,0,:] = 0.0
    mu_e[:,-1,:] = 0.0
    mu_e[:,:,0] = 0.0
    mu_e[:,:,-1] = 0.0

    return mu_e


# @nb.jit(nopython=True)
def comp_sponge_term(Q, Qref, sigma):
    """Compute sponge term
    decays the solution near boundaries to prevent undesired
    reflection of characteristics
    """
    return sigma * (Qref - Q)


def compE(Q, x, y, z, Rgas, mu, kappa, D, step):
    """Calculate x direction flux term"""
    _rho, _u, _v, _w, _p, _phi = ConsToPrim(Q, g.gamma)
    _temp = _p / (_rho * Rgas)

    tau_xx = Tauxx(_u, x, y, z, mu, step)
    tau_xy = Tauxy(_u, _v, x, y, z, mu, 0, step)
    tau_xz = Tauxz(_u, _w, x, y, z, mu, 0, step)
    qx = Qx(_temp, x, y, z, kappa, step)
    phix = Phix(_phi, x, y, z, D, step)

    g.E[:, :, :, 0] = _rho * _u
    g.E[:, :, :, 1] = _rho * _u**2.0 + _p - tau_xx
    g.E[:, :, :, 2] = _rho * _u * _v - tau_xy
    g.E[:, :, :, 3] = _rho * _u * _w - tau_xz
    g.E[:, :, :, 4] = (Q[:, :, :, 4] + _p) * _u - \
        _u * tau_xx - _v * tau_xy - _w * tau_xz + qx
    g.E[:, :, :, 5] = _rho * _u * _phi - phix


def compF(Q, x, y, z, Rgas, mu, kappa, D, step):
    """Calculate y direction flux term"""
    _rho, _u, _v, _w, _p, _phi = ConsToPrim(Q, g.gamma)
    _temp = _p / (_rho * Rgas)

    tau_yy = Tauyy(_v, x, y, z, mu, step)
    tau_xy = Tauxy(_u, _v, x, y, z, mu, 1, step)
    tau_yz = Tauyz(_v, _w, x, y, z, mu, 1, step)
    qy = Qy(_temp, x, y, z, kappa, step)
    phiy = Phiy(_phi, x, y, z, D, step)

    g.F[:, :, :, 0] = _rho * _v
    g.F[:, :, :, 1] = _rho * _u * _v - tau_xy
    g.F[:, :, :, 2] = _rho * _v**2.0 + _p - tau_yy
    g.F[:, :, :, 3] = _rho * _v * _w - tau_yz
    g.F[:, :, :, 4] = (Q[:, :, :, 4] + _p) * _v - \
        _u * tau_xy - _v * tau_yy - _w * tau_yz + qy
    g.F[:, :, :, 5] = _rho * _v * _phi - phiy


def compG(Q, x, y, z, Rgas, mu, kappa, D, step):
    """Calculate z direction flux term"""
    _rho, _u, _v, _w, _p, _phi = ConsToPrim(Q, g.gamma)
    _temp = _p / (_rho * Rgas)

    tau_zz = Tauzz(_w, x, y, z, mu, step)
    tau_xz = Tauxz(_u, _w, x, y, z, mu, 2, step)
    tau_yz = Tauyz(_v, _w, x, y, z, mu, 2, step)
    qz = Qz(_temp, x, y, z, kappa, step)
    phiz = Phiz(_phi, x, y, z, D, step)

    g.G[:, :, :, 0] = _rho * _w
    g.G[:, :, :, 1] = _rho * _u * _w - tau_xz
    g.G[:, :, :, 2] = _rho * _v * _w - tau_yz
    g.G[:, :, :, 3] = _rho * _w**2.0 + _p - tau_zz
    g.G[:, :, :, 4] = (Q[:, :, :, 4] + _p) * _w - \
        _u * tau_xz - _v * tau_yz - _w * tau_zz + qz
    g.G[:, :, :, 5] = _rho * _w * _phi - phiz


def compute_flux_limiter(Q, x, y, z):
    nx = Q.shape[0]
    ny = Q.shape[1]
    nz = Q.shape[2]
    nvars = Q.shape[3]

    # Part 1: Calculate r's
    rxm = np.zeros((nx, ny, nz), dtype=np.float64)
    rxp = np.zeros((nx, ny, nz), dtype=np.float64)
    rym = np.zeros((nx, ny, nz), dtype=np.float64)
    ryp = np.zeros((nx, ny, nz), dtype=np.float64)
    rzm = np.zeros((nx, ny, nz), dtype=np.float64)
    rzp = np.zeros((nx, ny, nz), dtype=np.float64)
    # X-direction flux limiter
    dQxm = Q[1:-1,:,:,:] - Q[0:-2,:,:,:]
    dQxp = Q[2:,:,:,:] - Q[1:-1,:,:,:]
    numtr = np.sum(dQxp*dQxm,axis=3)
    dentr1 = np.sum(dQxm*dQxm,axis=3)
    dentr2 = np.sum(dQxp*dQxp,axis=3)
    rzm[1:-1,:,:] = numtr / np.where(dentr1 > 1e-8, dentr1, np.inf)
    rzp[1:-1,:,:] = numtr / np.where(dentr2 > 1e-8, dentr2, np.inf)
    rxm[-1,:,:] = rxm[-2,:,:]
    rxp[-1,:,:] = rxp[-2,:,:]
    rxm[0,:,:] = rxm[1,:,:]
    rxp[0,:,:] = rxp[1,:,:]
    # Y-direction flux limiter
    dQym = Q[:,1:-1,:,:] - Q[:,0:-2,:,:]
    dQyp = Q[:,2:,:,:] - Q[:,1:-1,:,:]
    numtr = np.sum(dQyp*dQym,axis=3)
    dentr1 = np.sum(dQym*dQym,axis=3)
    dentr2 = np.sum(dQyp*dQyp,axis=3)
    rzm[:,1:-1,:] = numtr / np.where(dentr1 > 1e-8, dentr1, np.inf)
    rzp[:,1:-1,:] = numtr / np.where(dentr2 > 1e-8, dentr2, np.inf)
    rym[:,-1,:] = rym[:,-2,:]
    ryp[:,-1,:] = ryp[:,-2,:]
    rym[:,0,:] = rym[:,1,:]
    ryp[:,0,:] = ryp[:,1,:]
    # Z-direction flux limiter
    dQzm = Q[:,:,1:-1,:] - Q[:,:,0:-2,:]
    dQzp = Q[:,:,2:,:] - Q[:,:,1:-1,:]
    numtr = np.sum(dQzp*dQzm,axis=3)
    dentr1 = np.sum(dQzm*dQzm,axis=3)
    dentr2 = np.sum(dQzp*dQzp,axis=3)
    rzm[:,:,1:-1] = numtr / np.where(dentr1 > 1e-8, dentr1, np.inf)
    rzp[:,:,1:-1] = numtr / np.where(dentr2 > 1e-8, dentr2, np.inf)
    rzm[:,:,-1] = rzm[:,:,-2]
    rzp[:,:,-1] = rzp[:,:,-2]
    rzm[:,:,0] = rzm[:,:,1]
    rzp[:,:,0] = rzp[:,:,1]
    # communicate internal planes
    bc.communicate_internal_planes(rxm)
    bc.communicate_internal_planes(rxp)
    bc.communicate_internal_planes(rym)
    bc.communicate_internal_planes(ryp)
    bc.communicate_internal_planes(rzm)
    bc.communicate_internal_planes(rzp)
    
    # Part 2: Calculate Courant number
    dx = np.gradient(x,axis=0)
    dy = np.gradient(y,axis=1)
    dz = np.gradient(z,axis=2)
    rho, u, v, w, p, phi = ConsToPrim(Q, g.gamma)
    sos = np.sqrt(g.gamma*p/rho)
    Cr = g.dt*(np.abs(u)/dx + np.abs(v)/dy + np.abs(w)/dz + \
               sos*np.sqrt(1.0/dx**2.0 + 1.0/dy**2.0 + 1.0/dz**2.0))

    # Part 3: Calculate phi(r) flux limiter function
    phixm = np.minimum(2*np.where(rxm > 0, rxm, 0), 1)
    phixp = np.minimum(2*np.where(rxp > 0, rxp, 0), 1)
    phiym = np.minimum(2*np.where(rym > 0, rym, 0), 1)
    phiyp = np.minimum(2*np.where(ryp > 0, ryp, 0), 1)
    phizm = np.minimum(2*np.where(rzm > 0, rzm, 0), 1)
    phizp = np.minimum(2*np.where(rzp > 0, rzp, 0), 1)

    # Part 4: Calculate K's
    CoCr = np.where(Cr > 0.5, 0.25, Cr*(1-Cr))
    Kxm = np.zeros((nx,ny,nz,1),dtype=np.float64)
    Kxp = np.zeros((nx,ny,nz,1),dtype=np.float64)
    Kym = np.zeros((nx,ny,nz,1),dtype=np.float64)
    Kyp = np.zeros((nx,ny,nz,1),dtype=np.float64)
    Kzm = np.zeros((nx,ny,nz,1),dtype=np.float64)
    Kzp = np.zeros((nx,ny,nz,1),dtype=np.float64)
    Kxm[:,:,:,0] = 0.5*CoCr*(1.0-phixm)
    Kxp[:,:,:,0] = 0.5*CoCr*(1.0-phixp)
    Kym[:,:,:,0] = 0.5*CoCr*(1.0-phiym)
    Kyp[:,:,:,0] = 0.5*CoCr*(1.0-phiyp)
    Kzm[:,:,:,0] = 0.5*CoCr*(1.0-phizm)
    Kzp[:,:,:,0] = 0.5*CoCr*(1.0-phizp)

    # Part 5: Compute D's
    Dxm = np.zeros((nx,ny,nz,nvars),dtype=np.float64)
    Dxp = np.zeros((nx,ny,nz,nvars),dtype=np.float64)
    Dym = np.zeros((nx,ny,nz,nvars),dtype=np.float64)
    Dyp = np.zeros((nx,ny,nz,nvars),dtype=np.float64)
    Dzm = np.zeros((nx,ny,nz,nvars),dtype=np.float64)
    Dzp = np.zeros((nx,ny,nz,nvars),dtype=np.float64)
    
    Dxm[1:nx,:,:,:] = (Kxp[0:nx-1,:,:] + Kxm[1:nx,:,:])*(Q[1:nx,:,:,:]-Q[0:nx-1,:,:,:])
    Dxm[0,:,:,:] = Dxm[1,:,:,:]
    Dxp[0:nx-1,:,:,:] = (Kxp[0:nx-1,:,:] + Kxm[1:nx,:,:])*(Q[1:nx,:,:,:]-Q[0:nx-1,:,:,:])
    Dxp[nx-1,:,:,:] = Dxp[nx-2,:,:,:]
    Dym[:,1:ny,:,:] = (Kyp[:,0:ny-1,:] + Kym[:,1:ny,:])*(Q[:,1:ny,:,:]-Q[:,0:ny-1,:,:])
    Dym[:,0,:,:] = Dym[:,1,:,:]
    Dyp[:,0:ny-1,:,:] = (Kyp[:,0:ny-1,:] + Kym[:,1:ny,:])*(Q[:,1:ny,:,:]-Q[:,0:ny-1,:,:])
    Dyp[:,ny-1,:,:] = Dyp[:,ny-2,:,:]
    Dzm[:,:,1:nz,:] = (Kzp[:,:,0:nz-1] + Kzm[:,:,1:nz])*(Q[:,:,1:nz,:]-Q[:,:,0:nz-1,:])
    Dzm[:,:,0,:] = Dzm[:,:,1,:]
    Dzp[:,:,0:nz-1,:] = (Kzp[:,:,0:nz-1] + Kzm[:,:,1:nz])*(Q[:,:,1:nz,:]-Q[:,:,0:nz-1,:])
    Dzp[:,:,nz-1,:] = Dzp[:,:,nz-2,:]
    
    bc.communicate_internal_planes(Dxm)
    bc.communicate_internal_planes(Dxp)
    bc.communicate_internal_planes(Dym)
    bc.communicate_internal_planes(Dyp)
    bc.communicate_internal_planes(Dzm)
    bc.communicate_internal_planes(Dzp)
    
    # Part 6: Return RHS forcing term
    rhs = Dxp - Dxm + Dyp - Dym + Dzp - Dzm
    return rhs


def compRHS(Q, x, y, z, step):
    """Compute RHS flux and forcing terms

    Alternate differencing direction every step, e.g.
     - predictor step -> forward differences
     - corrector step -> backward differences

     - opposite differencing direction for aligned flux terms
     - central differencing for non-aligned flux terms
    """
    # Check bits for differentiation direction
    xdir_id = (step >> 0) & 0b001
    ydir_id = (step >> 1) & 0b001
    zdir_id = (step >> 2) & 0b001

    compE(Q, x, y, z, g.R_g, g.mu + g.mu_sgs, g.k, g.D, xdir_id)
    compF(Q, x, y, z, g.R_g, g.mu + g.mu_sgs, g.k, g.D, ydir_id)
    compG(Q, x, y, z, g.R_g, g.mu + g.mu_sgs, g.k, g.D, zdir_id)

    sponge_rhs = comp_sponge_term(Q, g.Qref, g.sponge_fac)

    for i in range(g.NVARS):
        g.dEdx[:, :, :, i] = compute_x_deriv(g.E[:, :, :, i],
                                             x, y, z, xdir_id)
        g.dFdy[:, :, :, i] = compute_y_deriv(g.F[:, :, :, i],
                                             x, y, z, ydir_id)
        g.dGdz[:, :, :, i] = compute_z_deriv(g.G[:, :, :, i],
                                             x, y, z, zdir_id)

    rhs = -1.0 * (g.dEdx + g.dFdy + g.dGdz) + sponge_rhs

    return rhs


#

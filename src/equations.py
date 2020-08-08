"""equations.py

Description:
 - Contains the governing equations of the flow and forcing terms.
 - Contains the numerical operators for computing derivatives.
 - Contains functions to convert conserved and primitive variables.
"""

# import numba as nb
import numpy as np

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
def ConsToPrim(Q):
    """ConsToPrim - Conserved variables to primitive variables
    Input:
     - Q : Vector of conserved variables (nx, ny, nz, nvars)
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
    _p = (g.gamma - 1) * (Q[:, :, :, 4] - 0.5 / Q[:, :, :, 0] *
                          (Q[:, :, :, 1] + Q[:, :, :, 2] + Q[:, :, :, 3])**2.0)
    _phi = Q[:, :, :, 5] / Q[:, :, :, 0]

    return _rho, _u, _v, _w, _p, _phi


# @nb.jit(nopython=True)
def PrimToCons(rho, u, v, w, p, phi):
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
    _e_tot = p / (g.gamma - 1.0) + 0.5 * rho * (u**2.0 + v**2.0 + w**2.0)
    _rho_phi = rho * phi

    return rho, _rho_u, _rho_v, _rho_w, _e_tot, _rho_phi


def Tauxx(U, x, y, z, mu, step):
    """Calculate stress tensor component xx"""
    if step == 'predictor':
        tau_xx = 2.0 * mu * compute_x_deriv(U, x, y, z, 1)
    elif step == 'corrector':
        tau_xx = 2.0 * mu * compute_x_deriv(U, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_xx


def Tauyy(V, x, y, z, mu, step):
    """Calculate stress tensor component yy"""
    if step == 'predictor':
        tau_yy = 2.0 * mu * compute_y_deriv(V, x, y, z, 1)
    elif step == 'corrector':
        tau_yy = 2.0 * mu * compute_y_deriv(V, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_yy


def Tauzz(W, x, y, z, mu, step):
    """Calculate stress tensor component zz"""
    if step == 'predictor':
        tau_zz = 2.0 * mu * compute_z_deriv(W, x, y, z, 1)
    elif step == 'corrector':
        tau_zz = 2.0 * mu * compute_z_deriv(W, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_zz


def Qx(T, x, y, z, k, step):
    """Calculate temperature diffusion in x"""
    if step == 'predictor':
        qx = -1.0 * k * compute_x_deriv(T, x, y, z, 1)
    elif step == 'corrector':
        qx = -1.0 * k * compute_x_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qx


def Qy(T, x, y, z, k, step):
    """Calculate temperature diffusion in y"""
    if step == 'predictor':
        qy = -1.0 * k * compute_y_deriv(T, x, y, z, 1)
    elif step == 'corrector':
        qy = -1.0 * k * compute_y_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qy


def Qz(T, x, y, z, k, step):
    """Calculate temperature diffusion in z"""
    if step == 'predictor':
        qz = -1.0 * k * compute_z_deriv(T, x, y, z, 1)
    elif step == 'corrector':
        qz = -1.0 * k * compute_z_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qz


def Phix(Phi, x, y, z, D, step):
    """Calculate scalar diffusion in x"""
    if step == 'predictor':
        phix = D * compute_x_deriv(Phi, x, y, z, 1)
    elif step == 'corrector':
        phix = D * compute_x_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phix


def Phiy(Phi, x, y, z, D, step):
    """Calculate scalar diffusion in y"""
    if step == 'predictor':
        phiy = D * compute_y_deriv(Phi, x, y, z, 1)
    elif step == 'corrector':
        phiy = D * compute_y_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phiy


def Phiz(Phi, x, y, z, D, step):
    """Calculate scalar diffusion in z"""
    if step == 'predictor':
        phiz = D * compute_z_deriv(Phi, x, y, z, 1)
    elif step == 'corrector':
        phiz = D * compute_z_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phiz


def Tauxy(U, V, x, y, z, mu, flux_dir, step):
    """Calculate stress tensor component xy"""
    if step == 'predictor':
        if flux_dir == 0:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 2) +
                           compute_x_deriv(V, x, y, z, 1))
        elif flux_dir == 1:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 1) +
                           compute_x_deriv(V, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 'corrector':
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


def Tauxz(U, W, x, y, z, mu, flux_dir, step):
    """Calculate stress tensor component xz"""
    if step == 'predictor':
        if flux_dir == 0:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 2) +
                           compute_x_deriv(W, x, y, z, 1))
        elif flux_dir == 2:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 1) +
                           compute_x_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 'corrector':
        if flux_dir == 0:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 2) +
                           compute_x_deriv(W, x, y, z, 0))
        elif flux_dir == 2:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 0) +
                           compute_x_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    return tau_xz


def Tauyz(V, W, x, y, z, mu, flux_dir, step):
    """Calculate stress tensor component yz"""
    if step == 'predictor':
        if flux_dir == 1:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 2) +
                           compute_y_deriv(W, x, y, z, 1))
        elif flux_dir == 2:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 1) +
                           compute_y_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 'corrector':
        if flux_dir == 1:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 2) +
                           compute_y_deriv(W, x, y, z, 0))
        elif flux_dir == 2:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 0) +
                           compute_y_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    return tau_yz


def comp_sponge_term(Q, Qref, sigma):
    """Compute sponge term
    decays the solution near boundaries to prevent undesired
    reflection of characteristics
    """
    return sigma * (Qref - Q)


def compE(Q, x, y, z, Rgas, mu, kappa, D, step):
    """Calculate x direction flux term"""
    _rho, _u, _v, _w, _p, _phi = ConsToPrim(Q)
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
    _rho, _u, _v, _w, _p, _phi = ConsToPrim(Q)
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
    _rho, _u, _v, _w, _p, _phi = ConsToPrim(Q)
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


def compRHS(Q, x, y, z, step):
    """Compute RHS flux and forcing terms

    Alternate differencing direction every step, e.g.
     - predictor step -> forward differences
     - corrector step -> backward differences

     - opposite differencing direction for aligned flux terms
     - central differencing for non-aligned flux terms
    """
    compE(Q, x, y, z, g.R_g, g.mu, g.k, g.D, step)
    compF(Q, x, y, z, g.R_g, g.mu, g.k, g.D, step)
    compG(Q, x, y, z, g.R_g, g.mu, g.k, g.D, step)

    sponge_rhs = comp_sponge_term(Q, g.Qref, g.sponge_fac)

    dir_id = None
    if step == 'predictor':
        dir_id = 0
    elif step == 'corrector':
        dir_id = 1
    else:
        raise Exception('Invalid derivative dir_id')

    for i in range(g.NVARS):
        g.dEdx[:, :, :, i] = compute_x_deriv(g.E[:, :, :, i],
                                             x, y, z, dir_id)
        g.dFdy[:, :, :, i] = compute_y_deriv(g.F[:, :, :, i],
                                             x, y, z, dir_id)
        g.dGdz[:, :, :, i] = compute_z_deriv(g.G[:, :, :, i],
                                             x, y, z, dir_id)

    rhs = -1.0 * (g.dEdx + g.dFdy + g.dGdz) + sponge_rhs

    return rhs


#

import common as g
import numpy as np


# ---------------------------------------------------
# Compute Finite Difference in x direction
#
# Input:
#     phi    : quantity
#     x      : x-dir grid
#     y      : y-dir grid
#     z      : z-dir grid
#     dir_id : 0 - Forward, 1 - Backward, 2 - Central
# Output:
#     d(phi)/dy : same dimensions as phi
# ---------------------------------------------------
def compute_x_deriv(phi, x, y, z, dir_id):
    nx = x.shape[0]-1
    ny = y.shape[1]-1
    nz = z.shape[2]-1
    dphi = np.zeros((nx+1, ny+1, nz+1))
    if dir_id == 0:
        dphi[nx, :, :] = (phi[nx, :, :] - phi[nx-1, :, :]) / \
                         (x[nx, :, :] - x[nx-1, :, :])
        dphi[0:nx-1, :, :] = (phi[1:nx, :, :] - phi[0:nx-1, :, :]) / \
                             (x[1:nx, :, :] - x[0:nx-1, :, :])
    if dir_id == 1:
        dphi[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / \
                        (x[1, :, :] - x[0, :, :])
        dphi[1:nx, :, :] = (phi[1:nx, :, :] - phi[0:nx-1, :, :]) / \
                           (x[1:nx, :, :] - x[0:nx-1, :, :])
    if dir_id == 2:
        dphi[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / \
                        (x[1, :, :] - x[0, :, :])
        dphi[nx, :, :] = (phi[nx, :, :] - phi[nx-1, :, :]) / \
                         (x[nx, :, :] - x[nx-1, :, :])
        dphi[1:nx-1, :, :] = (phi[2:nx, :, :] - phi[0:nx-2, :, :]) / \
                             (x[2:nx, :, :] - x[0:nx-2, :, :])
    return dphi


# ---------------------------------------------------
# Compute Finite Difference in y direction
#
# Input:
#     phi    : quantity
#     x      : x-dir grid
#     y      : y-dir grid
#     z      : z-dir grid
#     dir_id : 0 - Forward, 1 - Backward, 2 - Central
# Output:
#     d(phi)/dy : same dimensions as phi
# ---------------------------------------------------
def compute_y_deriv(phi, x, y, z, dir_id):
    nx = x.shape[0]-1
    ny = y.shape[1]-1
    nz = z.shape[2]-1
    dphi = np.zeros((nx+1, ny+1, nz+1))
    if dir_id == 0:
        dphi[:,     ny, :] = (phi[:,   ny, :] - phi[:,   ny-1, :]) / \
                             (y[:,     ny, :] - y[:,     ny-1, :])
        dphi[:, 0:ny-1, :] = (phi[:, 1:ny, :] - phi[:, 0:ny-1, :]) / \
                             (y[:,   1:ny, :] - y[:,   0:ny-1, :])
    if dir_id == 1:
        dphi[:,      0, :] = (phi[:,    1, :] - phi[:,      0, :]) / \
                             (y[:,      1, :] - y[:,        0, :])
        dphi[:,   1:ny, :] = (phi[:, 1:ny, :] - phi[:, 0:ny-1, :]) / \
                             (y[:,   1:ny, :] - y[:,   0:ny-1, :])
    if dir_id == 2:
        dphi[:,      0, :] = (phi[:,    1, :] - phi[:,      0, :]) / \
                             (y[:,      1, :] - y[:,        0, :])
        dphi[:,     ny, :] = (phi[:,   ny, :] - phi[:,   ny-1, :]) / \
                             (y[:,     ny, :] - y[:,     ny-1, :])
        dphi[:, 1:ny-1, :] = (phi[:, 2:ny, :] - phi[:, 0:ny-2, :]) / \
                             (y[:,   2:ny, :] - y[:,   0:ny-2, :])
    return dphi


# ---------------------------------------------------
# Compute Finite Difference in z direction
#
# Input:
#     phi    : quantity
#     x      : x-dir grid
#     y      : y-dir grid
#     z      : z-dir grid
#     dir_id : 0 - Forward, 1 - Backward, 2 - Central
# Output:
#     d(phi)/dz : same dimensions as phi
# ---------------------------------------------------
def compute_z_deriv(phi, x, y, z, dir_id):
    nx = x.shape[0]-1
    ny = y.shape[1]-1
    nz = z.shape[2]-1
    dphi = np.zeros((nx+1, ny+1, nz+1))
    if dir_id == 0:
        dphi[:, :,     nz] = (phi[:, :,   nz] - phi[:, :,   nz-1]) / \
                             (z[:, :,     nz] - z[:, :,     nz-1])
        dphi[:, :, 0:nz-1] = (phi[:, :, 1:nz] - phi[:, :, 0:nz-1]) / \
                             (z[:, :,   1:nz] - z[:, :,   0:nz-1])
    if dir_id == 1:
        dphi[:, :,      0] = (phi[:, :,    1] - phi[:, :,      0]) / \
                             (z[:, :,      1] - z[:, :,        0])
        dphi[:, :,   1:nz] = (phi[:, :, 1:nz] - phi[:, :, 0:nz-1]) / \
                             (z[:, :,   1:nz] - z[:, :,   0:nz-1])
    if dir_id == 2:
        dphi[:, :,      0] = (phi[:, :,    1] - phi[:, :,      0]) / \
                             (z[:, :,      1] - z[:, :,        0])
        dphi[:, :,     nz] = (phi[:, :,   nz] - phi[:, :,   nz-1]) / \
                             (z[:, :,     nz] - z[:, :,     nz-1])
        dphi[:, :, 1:nz-1] = (phi[:, :, 2:nz] - phi[:, :, 0:nz-2]) / \
                             (z[:, :,   2:nz] - z[:, :,   0:nz-2])
    return dphi


def ConsToPrim(Q):
    Rho_ = np.squeeze(Q[:, :, :, 0])
    U_ = np.squeeze(Q[:, :, :, 1] / Q[:, :, :, 0])
    V_ = np.squeeze(Q[:, :, :, 2] / Q[:, :, :, 0])
    W_ = np.squeeze(Q[:, :, :, 3] / Q[:, :, :, 0])
    P_ = np.squeeze((g.gamma - 1) * (Q[:, :, :, 4] - 0.5 / Q[:, :, :, 0] *
                    (Q[:, :, :, 1] + Q[:, :, :, 2] + Q[:, :, :, 3])**2))
    Phi_ = np.squeeze(Q[:, :, :, 5])

    return Rho_, U_, V_, W_, P_, Phi_


def PrimToCons(Rho, U, V, W, P, Phi):
    rhoU_ = Rho * U
    rhoV_ = Rho * V
    rhoW_ = Rho * W
    Et_ = P / (g.gamma - 1) + 0.5 * Rho * (U**2 + V**2 + W**2)
    rhoPhi_ = Rho * Phi

    return Rho, rhoU_, rhoV_, rhoW_, Et_, rhoPhi_


def Tauxx(U, x, y, z, mu, step):

    if step == 'predictor':
        tau_xx = 2 * mu * compute_x_deriv(U, x, y, z, 1)
    elif step == 'corrector':
        tau_xx = 2 * mu * compute_x_deriv(U, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_xx


def Tauyy(V, x, y, z, mu, step):

    if step == 'predictor':
        tau_yy = 2 * mu * compute_y_deriv(V, x, y, z, 1)
    elif step == 'corrector':
        tau_yy = 2 * mu * compute_y_deriv(V, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_yy


def Tauzz(W, x, y, z, mu, step):

    if step == 'predictor':
        tau_zz = 2 * mu * compute_z_deriv(W, x, y, z, 1)
    elif step == 'corrector':
        tau_zz = 2 * mu * compute_z_deriv(W, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return tau_zz


def Qx(T, x, y, z, k, step):

    if step == 'predictor':
        qx = -1 * k * compute_x_deriv(T, x, y, z, 1)
    elif step == 'corrector':
        qx = -1 * k * compute_x_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qx


def Qy(T, x, y, z, k, step):

    if step == 'predictor':
        qy = -1 * k * compute_y_deriv(T, x, y, z, 1)
    elif step == 'corrector':
        qy = -1 * k * compute_y_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qy


def Qz(T, x, y, z, k, step):

    if step == 'predictor':
        qz = -1 * k * compute_z_deriv(T, x, y, z, 1)
    elif step == 'corrector':
        qz = -1 * k * compute_z_deriv(T, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return qz


def Phix(Phi, x, y, z, D, step):

    if step == 'predictor':
        phix = D * compute_x_deriv(Phi, x, y, z, 1)
    elif step == 'corrector':
        phix = D * compute_x_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phix


def Phiy(Phi, x, y, z, D, step):

    if step == 'predictor':
        phiy = D * compute_y_deriv(Phi, x, y, z, 1)
    elif step == 'corrector':
        phiy = D * compute_y_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phiy


def Phiz(Phi, x, y, z, D, step):

    if step == 'predictor':
        phiz = D * compute_z_deriv(Phi, x, y, z, 1)
    elif step == 'corrector':
        phiz = D * compute_z_deriv(Phi, x, y, z, 0)
    else:
        raise Exception('Invalid Step')

    return phiz


def Tauxy(U, V, x, y, z, mu, flux_dir, step):

    if step == 'predictor':
        if flux_dir == 0:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 2) +
                           compute_x_deriv(V, x, y, z, 1))
        elif flux_dir == 1:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 2) +
                           compute_x_deriv(V, x, y, z, 0))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 'corrector':
        if flux_dir == 0:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 1) +
                           compute_x_deriv(V, x, y, z, 2))
        elif flux_dir == 1:
            tau_xy = mu * (compute_y_deriv(U, x, y, z, 0) +
                           compute_x_deriv(V, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    else:
        raise Exception('Invalid Step')

    return tau_xy


def Tauxz(U, W, x, y, z, mu, flux_dir, step):

    if step == 'predictor':
        if flux_dir == 0:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 2) +
                           compute_x_deriv(W, x, y, z, 1))
        elif flux_dir == 2:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 2) +
                           compute_x_deriv(W, x, y, z, 0))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 'corrector':
        if flux_dir == 0:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 1) +
                           compute_x_deriv(W, x, y, z, 2))
        elif flux_dir == 2:
            tau_xz = mu * (compute_z_deriv(U, x, y, z, 0) +
                           compute_x_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    return tau_xz


def Tauyz(V, W, x, y, z, mu, flux_dir, step):

    if step == 'predictor':
        if flux_dir == 1:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 2) +
                           compute_y_deriv(W, x, y, z, 1))
        elif flux_dir == 2:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 2) +
                           compute_x_deriv(W, x, y, z, 0))
        else:
            raise Exception('Invalid Flux Direction')

    elif step == 'corrector':
        if flux_dir == 1:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 1) +
                           compute_x_deriv(W, x, y, z, 2))
        elif flux_dir == 2:
            tau_yz = mu * (compute_z_deriv(V, x, y, z, 0) +
                           compute_x_deriv(W, x, y, z, 2))
        else:
            raise Exception('Invalid Flux Direction')

    return tau_yz


# -----------------------------------------------------
#  Sponge term
#   - decays the solution near boundaries to
#     prevent undesired reflection of characteristics
# -----------------------------------------------------
def comp_sponge_term(Q, Qref, sigma):
    return sigma * (Qref - Q)
# -----------------------------------------------------


def compE(Q, x, y, z, Rgas, mu, kappa, D, gamma, step):

    Rho, U, V, W, P, Phi = ConsToPrim(Q)

    T = P / (Rho * Rgas)

    tau_xx = Tauxx(U, x, y, z, mu, step)
    tau_xy = Tauxy(U, V, x, y, z, mu, 0, step)
    tau_xz = Tauxz(U, W, x, y, z, mu, 0, step)
    qx = Qx(T, x, y, z, kappa, step)
    phix = Phix(Phi, x, y, z, D, step)

    g.E[:, :, :, 0] = Rho * U
    g.E[:, :, :, 1] = Rho * U**2 + P - tau_xx
    g.E[:, :, :, 2] = Rho * U * V - tau_xy
    g.E[:, :, :, 3] = Rho * U * W - tau_xz
    g.E[:, :, :, 4] = (Q[:, :, :, 4] + P) * U - \
        U * tau_xx - V * tau_xy - W * tau_xz + qx
    g.E[:, :, :, 5] = Rho * U * Phi - phix


def compF(Q, x, y, z, Rgas, mu, kappa, D, gamma, step):

    Rho, U, V, W, P, Phi = ConsToPrim(Q)

    T = P / (Rho * Rgas)

    tau_yy = Tauyy(V, x, y, z, mu, step)
    tau_xy = Tauxy(U, V, x, y, z, mu, 1, step)
    tau_yz = Tauyz(V, W, x, y, z, mu, 1, step)
    qy = Qy(T, x, y, z, kappa, step)
    phiy = Phiy(Phi, x, y, z, D, step)

    g.F[:, :, :, 0] = Rho * V
    g.F[:, :, :, 1] = Rho * U * V - tau_xy
    g.F[:, :, :, 2] = Rho * V**2 + P - tau_yy
    g.F[:, :, :, 3] = Rho * V * W - tau_yz
    g.F[:, :, :, 4] = (Q[:, :, :, 4] + P) * V - \
        U * tau_xy - V * tau_yy - W * tau_yz + qy
    g.F[:, :, :, 5] = Rho * V * Phi - phiy


def compG(Q, x, y, z, Rgas, mu, kappa, D, gamma, step):

    Rho, U, V, W, P, Phi = ConsToPrim(Q)

    T = P / (Rho * Rgas)

    tau_zz = Tauzz(W, x, y, z, mu, step)
    tau_xz = Tauxz(U, W, x, y, z, mu, 2, step)
    tau_yz = Tauyz(V, W, x, y, z, mu, 2, step)
    qz = Qz(T, x, y, z, kappa, step)
    phiz = Phiz(Phi, x, y, z, D, step)

    g.G[:, :, :, 0] = Rho * W
    g.G[:, :, :, 1] = Rho * U * W - tau_xz
    g.G[:, :, :, 2] = Rho * V * W - tau_yz
    g.G[:, :, :, 3] = Rho * W**2 + P - tau_zz
    g.G[:, :, :, 4] = (Q[:, :, :, 4] + P) * W - \
        U * tau_xz - V * tau_yz - W * tau_zz + qz
    g.G[:, :, :, 5] = Rho * W * Phi - phiz


def compRHS(Q, x, y, z, step):

    compE(Q, x, y, z, g.R_g, g.mu, g.k, g.D, g.gamma, step)
    compF(Q, x, y, z, g.R_g, g.mu, g.k, g.D, g.gamma, step)
    compG(Q, x, y, z, g.R_g, g.mu, g.k, g.D, g.gamma, step)

    sponge_rhs = comp_sponge_term(Q, g.Qref, g.sponge_fac)

    dir_id = None

    if step == 'predictor':
        dir_id = 0
    elif step == 'corrector':
        dir_id = 1
    else:
        raise Exception('Invalid derivative dir_id')

    for ii in range(g.NVARS):
        g.dEdx[:, :, :, ii] = compute_x_deriv(g.E[:, :, :, ii],
                                              x, y, z, dir_id)
        g.dFdy[:, :, :, ii] = compute_y_deriv(g.F[:, :, :, ii],
                                              x, y, z, dir_id)
        g.dGdz[:, :, :, ii] = compute_z_deriv(g.G[:, :, :, ii],
                                              x, y, z, dir_id)

    rhs = -1 * (g.dEdx + g.dFdy + g.dGdz) + sponge_rhs

    return rhs


#

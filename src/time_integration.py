import numpy as np

import common as g
import equations as eq
import boundary_conditions as bc


def compute_timestep_maccormack():
    """Advance time step using MacCormack scheme time advancement"""

    # Update simulation time
    g.t = g.t + g.dt
    # Save state from previous time step
    g.Qo[:, :, :, :] = g.Q[:, :, :, :]

    # Compute first RK step
    k1 = eq.compRHS(g.Q, g.xg, g.yg, g.zg, g.rk_step_1)
    g.Q[:, :, :, :] = g.Q[:, :, :, :] + g.dt*k1[:, :, :, :]

    bc.apply_boundary_conditions()

    # Compute second RK step
    k2 = eq.compRHS(g.Q, g.xg, g.yg, g.zg, g.rk_step_2)
    g.Q[:, :, :, :] = g.Qo[:, :, :, :] + \
        g.dt*(k1[:, :, :, :] + k2[:, :, :, :])/2.0

    bc.apply_boundary_conditions()

    # Switch differentiation directions between time steps
    g.rk_step_1, g.rk_step_2 = g.rk_step_2, g.rk_step_1


def compute_dt():
    """Compute time step size based on CFL"""

    Rho_, U_, V_, W_, P_, _ = eq.ConsToPrim(g.Q)
    a0 = np.sqrt(g.gamma*P_ / Rho_)

    Ur = np.abs(U_ + a0)
    Ul = np.abs(U_ - a0)
    U_ = np.maximum(Ur, Ul)

    Vr = np.abs(V_ + a0)
    Vl = np.abs(V_ - a0)
    V_ = np.maximum(Vr, Vl)

    Wr = np.abs(W_ + a0)
    Wl = np.abs(W_ - a0)
    W_ = np.maximum(Wr, Wl)

    dx = np.gradient(g.xg, axis=0)
    dy = np.gradient(g.yg, axis=1)
    dz = np.gradient(g.zg, axis=2)

    # Calculate dt from CFL
    dt = g.CFL_ref / (U_/dx + V_/dy + W_/dz)

    # MPI buffers
    dt_local = np.empty((1), dtype=np.float64)
    dt_global = np.empty((1), dtype=np.float64)

    # Find the minimum dt across processes
    dt_local[0] = np.min(dt)
    g.comm.Reduce(dt_local[0], dt_global[0], op=g.MPI.MIN, root=0)

    # broadcast time step size
    g.dt = g.comm.Bcast(dt_global[0], root=0)


#

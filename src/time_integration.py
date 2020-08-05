"""time_integration.py
Compute time step advancement.

Contains MacCormack time step implementation.
"""

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
    kv1 = eq.compRHS(g.Q, g.xg, g.yg, g.zg, g.rk_step_1)
    g.Q[:, :, :, :] = g.Q[:, :, :, :] + g.dt*kv1[:, :, :, :]

    bc.apply_boundary_conditions()

    # Compute second RK step
    kv2 = eq.compRHS(g.Q, g.xg, g.yg, g.zg, g.rk_step_2)
    g.Q[:, :, :, :] = g.Qo[:, :, :, :] + \
        g.dt*(kv1[:, :, :, :] + kv2[:, :, :, :])/2.0

    bc.apply_boundary_conditions()

    # Switch differentiation directions between time steps
    g.rk_step_1, g.rk_step_2 = g.rk_step_2, g.rk_step_1


def compute_dt():
    """Compute time step size based on CFL"""

    _rho, _u, _v, _w, _p, _ = eq.ConsToPrim(g.Q)
    _a0 = np.sqrt(g.gamma*_p / _rho)

    _ur = np.abs(_u + _a0)
    _ul = np.abs(_u - _a0)
    _u = np.maximum(_ur, _ul)

    _vr = np.abs(_v + _a0)
    _vl = np.abs(_v - _a0)
    _v = np.maximum(_vr, _vl)

    _wr = np.abs(_w + _a0)
    _wl = np.abs(_w - _a0)
    _w = np.maximum(_wr, _wl)

    _dx = np.gradient(g.xg, axis=0)
    _dy = np.gradient(g.yg, axis=1)
    _dz = np.gradient(g.zg, axis=2)

    # Calculate dt from CFL
    _dt = g.CFL_ref / (_u/_dx + _v/_dy + _w/_dz)

    # MPI buffers
    dt_local = np.empty((1), dtype=np.float64)
    dt_global = np.empty((1), dtype=np.float64)

    # Find the minimum dt across processes
    dt_local[0] = np.min(_dt)
    g.comm.Reduce(dt_local, dt_global, op=g.MPI.MIN, root=0)

    # broadcast time step size
    g.comm.Bcast(dt_global, root=0)
    g.dt = dt_global[0]


#

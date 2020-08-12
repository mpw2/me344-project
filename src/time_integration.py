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
    kv1 = eq.compRHS(g.Q, g.xg, g.yg, g.zg, (g.rk_step_bits))
    g.Q[:, :, :, :] = g.Q[:, :, :, :] + g.dt*kv1[:, :, :, :]

    bc.apply_boundary_conditions()

    # Compute second RK step
    kv2 = eq.compRHS(g.Q, g.xg, g.yg, g.zg, (g.rk_step_bits ^ 0b111))
    
    g.Q[:, :, :, :] = 0.5*(g.Qo[:, :, :, :] + g.Q[:, :, :, :] + \
                           g.dt*kv2[:, :, :, :])
    
    # Compute and apply flux limiter
    flux_lim = eq.compute_flux_limiter(g.Qo, g.xg, g.yg, g.zg)
    g.Q[:, :, :, :] = g.Q[:, :, :, :] + flux_lim
    
    bc.apply_boundary_conditions()

    # Switch differentiation directions between time steps
    # Use round robin style to permute direction combinations
    g.rk_step_bits = g.rk_step_bits + 0b001

    g.comm.Barrier()

def compute_dt():
    """Compute time step size based on CFL"""

    _rho, _u, _v, _w, _p, _ = eq.ConsToPrim(g.Q, g.gamma)
    _sos = np.sqrt(g.gamma * _p / _rho)

    _u = np.abs(_u)
    _v = np.abs(_v)
    _w = np.abs(_w)

    _dx = np.gradient(g.xg, axis=0)
    _dy = np.gradient(g.yg, axis=1)
    _dz = np.gradient(g.zg, axis=2)

    _dixyz = 1.0/(_dx**2.0) + 1.0/(_dy**2.0) + 1.0/(_dz**2.0)
    
    _nuprime = (4.0 * (g.mu + g.mu_sgs)**2.0 * g.gamma) / (3.0 * g.Pr * _rho)

    _dt = _u / _dx + _v / _dy + _w / _dz + _sos * np.sqrt(_dixyz) + \
        2.0 * _nuprime * _dixyz
    _dt = 1.0 / _dt
    _dt = _dt * g.CFL_ref
    
    # MPI buffers
    dt_local = np.empty((1), dtype=np.float64)
    dt_global = np.empty((1), dtype=np.float64)

    # Find the minimum dt across processes
    dt_local[0] = np.min(_dt)
    g.comm.Reduce([dt_local, g.MPI.DOUBLE], [dt_global, g.MPI.DOUBLE],
                  op=g.MPI.MIN, root=0)

    # broadcast time step size
    g.comm.Bcast(dt_global, root=0)
    g.dt = dt_global[0]


#

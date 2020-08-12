"""monitor.py
Writes solver monitor output to std_out periodically.
"""

import sys
import numpy as np

import common as g
import equations as eq


def output_monitor():
    """Write monitor output to stdout"""

    # Check for NaNs
    if np.any(np.isnan(g.Q)):
        sys.stderr.write(
            "[PROC {0:d}] Error: NaNs! \n".format(g.myrank))
        sys.stderr.flush()
        sys.stdout.flush()
        g.comm.Abort()

    # Allocate MPI memory buffers
    sendbuf = np.empty((g.NVARS), dtype=np.float64)
    recvbuf = np.empty((g.NVARS), dtype=np.float64)

    # Compute primitive variables
    _rho, _u, _v, _w, _p, _phi = eq.ConsToPrim(g.Q, g.gamma)

    # Compute globally averaged variables
    rho_sum = np.sum(_rho)
    u_sum = np.sum(_u)
    v_sum = np.sum(_v)
    w_sum = np.sum(_w)
    p_sum = np.sum(_p)
    phi_sum = np.sum(_phi)

    sendbuf = np.array([rho_sum, u_sum, v_sum, w_sum, p_sum, phi_sum])
    g.comm.Reduce([sendbuf, g.MPI.DOUBLE], [recvbuf, g.MPI.DOUBLE],
                  op=g.MPI.SUM, root=0)
    recvbuf = recvbuf / (g.nx_global * g.ny_global * g.nz_global)
    rho_mean, u_mean, v_mean, w_mean, p_mean, phi_mean = recvbuf

    # Compute global max variables
    rho_max = np.max(_rho)
    u_max = np.max(_u)
    v_max = np.max(_v)
    w_max = np.max(_w)
    p_max = np.max(_p)
    phi_max = np.max(_phi)

    sendbuf = np.array([rho_max, u_max, v_max, w_max, p_max, phi_max])
    g.comm.Reduce([sendbuf, g.MPI.DOUBLE], [recvbuf, g.MPI.DOUBLE],
                  op=g.MPI.MAX, root=0)
    rho_max, u_max, v_max, w_max, p_max, phi_max = recvbuf

    # only print monitor if rank 0
    if g.myrank != 0:
        return

    print('---- monitor ----')
    print('time step : {:d}'.format(g.tstep))
    print('sim. time : {:.4e}'.format(g.t))
    print('')
    print(('Max  Rho, U, V, W, P, Phi : {0:10.4e}, {1:10.4e}, '
           '{2:10.4e}, {3:10.4e}, {4:10.4e}, {5:10.4e}'
           ).format(rho_max, u_max, v_max, w_max, p_max, phi_max))
    print(('Mean Rho, U, V, W, P, Phi : {0:10.4e}, {1:10.4e}, '
           '{2:10.4e}, {3:10.4e}, {4:10.4e}, {5:10.4e}'
           ).format(rho_mean, u_mean, v_mean, w_mean, p_mean, phi_mean))
    print('')
    sys.stdout.flush()


def output_final():
    """Print info upon program completion"""

    # only print if rank 0
    if g.myrank != 0:
        return

    print('Done!')
    sys.stdout.flush()


#

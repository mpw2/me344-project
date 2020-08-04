import numpy as np

import common as g
import equations as eq


def output_monitor():
    """Write monitor output to stdout"""

    # Check for NaNs
    checkQ = np.sum(g.Q, axis=(0, 1, 2))
    if np.any(np.isnan(checkQ)):
        raise Exception('Error: NaNs!!!')

    # Allocate MPI memory buffers
    sendbuf = np.array([g.NVARS], dtype=np.float64)
    recvbuf = np.array([g.NVARS], dtype=np.float64)

    # Compute primitive variables
    Rho, U, V, W, P, Phi = eq.ConsToPrim(g.Q)

    # Compute globally averaged variables
    Rho_sum = np.sum(Rho)
    U_sum = np.sum(U)
    V_sum = np.sum(V)
    W_sum = np.sum(W)
    P_sum = np.sum(P)
    Phi_sum = np.sum(Phi)

    sendbuf[:] = [Rho_sum, U_sum, V_sum, W_sum, P_sum, Phi_sum]
    g.comm.Reduce(sendbuf, recvbuf, op=g.MPI.MPI.SUM, root=0)
    recvbuf = recvbuf / (g.nx_global * g.ny_global * g.nz_global)
    Rho_mean, U_mean, V_mean, W_mean, P_mean, Phi_mean = recvbuf

    # Compute global max variables
    Rho_max = np.max(Rho)
    U_max = np.max(U)
    V_max = np.max(V)
    W_max = np.max(W)
    P_max = np.max(P)
    Phi_max = np.max(Phi)

    sendbuf[:] = [Rho_max, U_max, V_max, W_max, P_max, Phi_max]
    g.comm.Reduce(sendbuf, recvbuf, op=g.MPI.MAX, root=0)
    Rho_max, U_max, V_max, W_max, P_max, Phi_max = recvbuf

    # only print monitor if rank 0
    if g.myrank != 0:
        return

    print('---- monitor ----')
    print('time step : {:d}'.format(g.tstep))
    print('sim. time : {:.4e}'.format(g.t))
    print('')
    print(('Max  Rho, U, V, W, P, Phi : {0:10.4e}, {1:10.4e}, '
           '{2:10.4e}, {3:10.4e}, {4:10.4e}, {5:10.4e}'
           ).format(Rho_max, U_max, V_max, W_max, P_max, Phi_max))
    print(('Mean Rho, U, V, W, P, Phi : {0:10.4e}, {1:10.4e}, '
           '{2:10.4e}, {3:10.4e}, {4:10.4e}, {5:10.4e}'
           ).format(Rho_mean, U_mean, V_mean, W_mean, P_mean, Phi_mean))
    print('')


#

import numpy as np

import common as g
import equations as eq


def output_monitor():
    # ---------------------------------------------
    # Check for NaNs
    checkQ = np.sum(g.Q, axis=(0, 1, 2))
    if np.any(np.isnan(checkQ)):
        raise Exception('Error: NaNs!!!')

    # --------------------------------------------
    print('---- monitor ----')
    print('time step : {:d}'.format(g.tstep))
    print('sim. time : {:.4e}'.format(g.t))

    Rho, U, V, W, P, Phi = eq.ConsToPrim(g.Q)

    Rho_mean = np.mean(Rho)
    U_mean = np.mean(U)
    V_mean = np.mean(V)
    W_mean = np.mean(W)
    P_mean = np.mean(P)
    Phi_mean = np.mean(Phi)

    Rho_max = np.max(Rho)
    U_max = np.max(U)
    V_max = np.max(V)
    W_max = np.max(W)
    P_max = np.max(P)
    Phi_max = np.max(Phi)

    print('')

    print(('Max  Rho, U, V, W, P, Phi : {0:10.4e}, {1:10.4e}, '
           '{2:10.4e}, {3:10.4e}, {4:10.4e}, {5:10.4e}'
           ).format(Rho_max, U_max, V_max, W_max, P_max, Phi_max))
    print(('Mean Rho, U, V, W, P, Phi : {0:10.4e}, {1:10.4e}, '
           '{2:10.4e}, {3:10.4e}, {4:10.4e}, {5:10.4e}'
           ).format(Rho_mean, U_mean, V_mean, W_mean, P_mean, Phi_mean))

    print('')


#

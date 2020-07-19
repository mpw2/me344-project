import common as g
import equations as eq
import numpy as np

def output_monitor():
    
    print('---- monitor ----')
    print('time step : {:d}'.format(g.tstep))
    print('sim. time : {:.4e}'.format(g.t))
    
    Rho,U,V,W, P = eq.ConsToPrim(g.Q)
    Rho_mean = np.mean(Rho)
    U_mean = np.mean(U)
    V_mean = np.mean(V)
    W_mean = np.mean(W)
    P_mean = np.mean(P)
    Rho_max = np.max(Rho)
    U_max = np.max(U)
    V_max = np.max(V)
    W_max = np.max(W)
    P_max = np.max(P)
    
    print('')
    
    print('Max  Rho, U, V, W, P : {0:10.4e}, {1:10.4e}, {2:10.4e}, {3:10.4e}, {4:10.4e}'.format(Rho_max,U_max,V_max,W_max,P_max))
    print('Mean Rho, U, V, W, P : {0:10.4e}, {1:10.4e}, {2:10.4e}, {3:10.4e}, {3:10.4e}'.format(Rho_mean,U_mean,V_mean,W_mean,P_mean))
    
    print('')


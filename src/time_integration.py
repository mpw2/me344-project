import common as g
from equations import *
from boundary_conditions import *

def compute_timestep_maccormack():
    g.t = g.t + g.dt

    g.Qo[:,:,:] = g.Q[:,:,:]
    
    k1 = compRHS(g.Q, g.xg, g.yg, g.rk_step_1)
    g.Q[1:g.nx-1,1:g.ny-1,:] = g.Q[1:g.nx-1,1:g.ny-1,:] + g.dt*k1[1:g.nx-1,1:g.ny-1,:] 
    
    apply_boundary_conditions()

    k2 = compRHS(g.Q, g.xg, g.yg, g.rk_step_2)
    g.Q[1:g.nx-1,1:g.ny-1,:] = g.Qo[1:g.nx-1,1:g.ny-1,:] + g.dt*( k1[1:g.nx-1,1:g.ny-1,:] + k2[1:g.nx-1,1:g.ny-1,:] )/2.0
    
    apply_boundary_conditions() 

    g.rk_step_1, g.rk_step_2 = g.rk_step_2, g.rk_step_1


def compute_dt():
    
    Rho_,U_,V_,P_ = ConsToPrim(g.Q)
    a0 = np.sqrt( g.gamma*g.P/g.Rho )
    
    Ur = np.abs(U_ + a0)
    Ul = np.abs(U_ - a0)
    U_ = np.maximum(Ur,Ul)
    
    Vr = np.abs(V_ + a0)
    Vl = np.abs(V_ - a0)
    V_ = np.maximum(Vr,Vl)
    
    dx = np.gradient(g.xg, axis=0)
    dy = np.gradient(g.yg, axis=1)    
    
    dt = g.CFL_ref / (U_/dx + V_/dy)

    g.dt = np.min(dt)






import common as g
from equations import *
from boundary_conditions import *

def compute_timestep_maccormack():
    g.t = g.t + g.dt

    g.Qo[:,:,:,:] = g.Q[:,:,:,:]
    
    k1 = compRHS(g.Q,g.xg,g.yg,g.zg,g.rk_step_1)
    g.Q[:,:,:,:] = g.Q[:,:,:,:] + g.dt*k1[:,:,:,:] 
    
    apply_boundary_conditions()

    k2 = compRHS(g.Q,g.xg,g.yg,g.zg,g.rk_step_2)
    g.Q[:,:,:,:] = g.Qo[:,:,:,:] + g.dt*( k1[:,:,:,:] + k2[:,:,:,:] )/2.0
    
    apply_boundary_conditions() 

    g.rk_step_1, g.rk_step_2 = g.rk_step_2, g.rk_step_1


def compute_dt():
    
    Rho_,U_,V_,W_,P_ = ConsToPrim(g.Q)
    a0 = np.sqrt( g.gamma*g.P/g.Rho )
    
    Ur = np.abs(U_ + a0)
    Ul = np.abs(U_ - a0)
    U_ = np.maximum(Ur,Ul)
    
    Vr = np.abs(V_ + a0)
    Vl = np.abs(V_ - a0)
    V_ = np.maximum(Vr,Vl)

    Wr = np.abs(W_ + a0)
    Wl = np.abs(W_ - a0)
    W_ = np.maximum(Wr,Wl)
    
    dx = np.gradient(g.xg, axis=0)
    dy = np.gradient(g.yg, axis=1)    
    dz = np.gradient(g.zg, axis=2)
    
    dt = g.CFL_ref / (U_/dx + V_/dy + W_/dz)

    g.dt = np.min(dt)






import common as g
from equations import *
from boundary_conditions import *

def compute_timestep_maccormack():
    g.t = g.t + g.dt

    Qo = g.Q.copy()
    
    k1 = compRHS(g.Q,g.xg,g.yg,g.zg,'predictor')
    g.Q[1:g.nx-1,1:g.ny-1,1:g.nz-1,:] = g.Q[1:g.nx-1,1:g.ny-1,1:g.nz-1,:] + g.dt*k1[1:g.nx-1,1:g.ny-1,1:g.nz-1,:] 
    
    apply_boundary_conditions()

    k2 = compRHS(g.Q,g.xg,g.yg,g.zg,'corrector')
    g.Q[1:g.nx-1,1:g.ny-1,1:g.nz-1,:] = Qo[1:g.nx-1,1:g.ny-1,1:g.nz-1,:] + g.dt*( k1[1:g.nx-1,1:g.ny-1,1:g.nz-1,:] + k2[1:g.nx-1,1:g.ny-1,1:g.nz-1,:] )/2.0
    
    apply_boundary_conditions() 



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






from common import *

def compute_timestep_maccormack():
    Qo = Q.copy()
    
    k1 = compRHS(Q,x,y,'predictor')
    Q[1:nx-1,1:ny-1,:] = Q[1:nx-1,1:ny-1,:] + dt*k1[1:nx-1,1:ny-1,:] 
    
    apply_boundary_conditions()

    k2 = compRHS(Q,x,y,'corrector')
    Q[1:nx-1,1:ny-1,:]  = Qo[1:nx-1,1:ny-1,:] + dt*( k1[1:nx-1,1:ny-1,:] + k2[1:nx-1,1:ny-1,:] )/2.0
    
    apply_boundary_conditions() 



def compute_dt():
    # Output
    global dt
    
    Rho_,U_,V_,P_ = ConsToPrim(Q,gamma)
    a0 = np.sqrt( gamma*P/Rho )
    
    Ur = np.abs(U_ + a0)
    Ul = np.abs(U_ - a0)
    U_ = np.maximum(Ur,Ul)
    
    Vr = np.abs(V_ + a0)
    Vl = np.abs(V_ - a0)
    V_ = np.maximum(Vr,Vl)
     
    dx = np.gradient(xg)
    dy = np.gradient(yg)    
    
    dt = CFL_ref / (U_/dx + V_/dy)

    dt = np.min(dt)






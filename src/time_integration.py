
import global

def compute_timestep_maccormack():
    Qo = Q.copy()
    
    
    k1 = compRHS(Q,x,y,'predictor')
    Q[1:nx-1,1:ny-1,:] = Q[1:nx-1,1:ny-1,:] + dt*k1
    
    apply_boundary_conditions()

    k2 = compRHS(Q,x,y,'corrector')
    Q[1:nx-1,1:ny-1,:] = Qo[1:nx-1,1:ny-1,:] + dt*( k1 + k2 )/2.0
    
    apply_boundary_conditions() 



def compute_dt():
    # Output
    global dt
    
    Rho_,U_,V_,P_ = ConsToPrim(Q,gamma)
    Ucfl = np.sqrt( U_**2 + V_**2 )
    Ucfl = np.min(Ucfl)
    a0 = np.sqrt( gamma*P/Rho )
    Ucfl = Ucfl + a0
    dx = np.min( np.min( np.diff(xg) ), np.min( np.diff(yg) ) )
    dt = CFL_ref * dx / Ucfl






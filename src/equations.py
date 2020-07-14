import numpy as numpy
from global import xg
from global import nx 
from global import mu
from global import gamma
from global import nvars


# dir_id: ( 0 - Forward, 1 - Backward, 2 - Central )
# returns matrix of the same size as phi
def compute_x_deriv(phi, x, y, dir_id):
	nx = np.shape(x,0)
	ny = np.shape(y,1)
	lo_x = 1
	hi_x = nx-1
	lo_y = 1
	hi_y = ny-1
	dphi = np.zeros((nx,ny))
	if dir_id == 0:
		dphi[nx,:] = (phi[nx,:] - phi[nx-1,:]) / (x[nx,:] - x[nx-1,:])
		dphi[0:hi_x,:] = (phi[0+1:hi_x+1,:] - phi[0:hi_x, :]) / (x[0+1:hi_x+1,:] - x[0:hi_x,:])
	if dir_id = 1:
		dphi[0,:] = (phi[1,:] - phi[0,:]) / (x[1,:] - x[0,:])
		dphi[lo_x:nx,:] = (phi[lo_x:hi_x,:] - phi[lo_x-1:hi_x-1, :]) / (x[lo_x:hi_x,:] - x[lo_x-1:hi_x-1,:]) 
	if dir_id = 2:
		dphi[0,:] = (phi[1,:] - phi[0,:]) / (x[1,:] - x[0,:])
		dphi[nx,:] = (phi[nx,:] - phi[nx-1,:]) / (x[nx,:] - x[nx-1,:])
		dphi[lo_x:hi_x,:] = (phi[lo_x+1:hi_x+1, :] - phi[lo_x-1:hi_x-1, :]) / (x[lo_x+1:hi_x+1,:] - x[lo_x-1:hi_x-1,:]) 
	return dphi	

def compute_y_deriv(phi, x, y, dir_id):
	nx = np.shape(x,0)
	ny = np.shape(y,0)
	lo_x = 1
	hi_x = nx-1
	lo_y = 1
	hi_y = ny-1
	dphi = np.zeros((nx,ny))
	if dir_id == 0:
		dphi[:,ny] = (phi[:,ny] - phi[:,ny-1]) / (y[:,ny] - y[:,ny-1])
		dphi[:,0:hi_y] = (phi[:,0+1:hi_y+1] - phi[:, 0:hi_y]) / (y[:, 0:hi_y+1] - y[:, 0:hi_y]) 
	if dir_id = 1:
		dphi[:,0] = (phi[:,1] - phi[:,0]) / (y[:,1] - y[:,0])
		dphi[:,lo_y:ny] = (phi[:,lo_y:ny] - phi[:, lo_y-1:ny-1]) / (y[:, lo_y:ny] - y[:, lo_y-1:ny-1]) 
	if dir_id = 2:
		dphi[:,0] = (phi[:,1] - phi[:,0]) / (y[:,1] - y[:,0])
		dphi[:,ny] = (phi[:,ny] - phi[:,ny-1]) / (y[:,ny] - y[:,ny-1])
		dphi[:,lo_y:hi_y] = (phi[:, lo_y+1:hi_y+1] - phi[: lo_y+1:hi_y+1]) / (y[:, lo_y+1:hi_y+1] - y[:, lo_y-1:hi_y-1]) 
	return dphi	

def ConsToPrim(Q,gamma):
	Rho = Q[:,:,0]
	U = Q[:,:,1] / Q[:,:,0]
	V = Q[:,:,2] / Q[:,:,0]
	P = (gamma - 1) * (Q[:,:,3] - 1 / 2 / Q[:,:,0] * (Q[:,:,1] + Q[:,:,2])**2)

	return Rho, U, V, P

def PrimToCons(Rho,U,V,P):
	rhoU = Rho * U
	rhoV = Rho * V
	Et = P / (gamma - 1) + Rho / 2 * (U**2 + V**2)

	return Rho, rhoU, rhoV, Et

def Tauxx(U,x,y,mu,step):

	if step == 'predictor':
		tau_xx = 2 * mu * compute_x_deriv(U,x,y,1)

	elif step == 'corrector':
		tau_xx = 2 * mu * compute_x_deriv(U,x,y,0)

	else:
		raise Exception('Invalid Step')

	return tau_xx

def Tauyy(V,x,y,mu,step):

	if step == 'predictor':
		tau_yy = 2 * mu * compute_y_deriv(V,x,y,1)
	elif step == 'corrector':
		tau_yy = 2 * mu * compute_y_deriv(V,x,y,0)
	else:
		raise Exception('Invalid Step')

	return tau_yy

def Tauxy(U,V,x,y,mu,flux_dir,step):
	nx = np.shape(x,0)
	ny = np.shape(y,1)
	tau_xy = np.zeros((nx,ny))

	if step == 'predictor':
		if flux_dir == 0:
			tau_xy[1:nx-1] = mu * (compute_y_deriv(U,x,y,2) + compute_x_deriv(V,x,y,1))
		elif flux_dir == 1:
			tau_xy[1:nx-1] = mu * (compute_y_deriv(U,x,y,2) + compute_x_deriv(V,x,y,0))
		else:
			raise Exception('Invalid Flux Direction')

	elif step == 'corrector':

		if flux_dir == 0:
			tau_xy = mu * (compute_y_deriv(U,x,y,1) + compute_x_deriv(V,x,y,2))
		elif flux_dir == 1:
			tau_xy = mu * (compute_y_deriv(U,x,y,0) + compute_x_deriv(V,x,y,2))
		else:
			raise Exception('Invalid Flux Direction')

	else:
		raise Exception('Invalid Step')

	return tau_xy

def compE(Q,x,y,mu,gamma,step):
	nx = np.shape(x,0)
	ny = np.shape(y,1)
	E = np.zeros([nx,ny,nvars])

	Rho, U, V, P = ConsToPrim(Q,gamma)

	tau_xx = Tauxx(U,x,y,mu,step)
	tau_xy = Tauxy(U,V,x,y,mu,0,step)

	E[:,:,0] = Rho * U
	E[:,:,1] = Rho * U**2 + P - tau_xx
	E[:,:,2] = Rho * U * V - tau_xy
	E[:,:,3] = (Q[:,:,-1] + P) * U - U * tau_xx - V * tau_xy

	return E

def compF(Q,x,y,mu,gamma,step):
	nx = np.shape(x,0)
	ny = np.shape(y,0)
	F = np.zeros([nx,ny,nvars])

	Rho, U, V, P = ConsToPrim(Q,gamma)

	tau_yy = Tauyy(V,x,y,mu,step)
	tau_xy = Tauxy(U,V,x,y,mu,1,step)

	F[:,:,0] = Rho * V
	F[:,:,1] = Rho * U * V - tau_xy
	F[:,:,2] = Rho * V**2 + P - tau_yy
	F[:,:,3] = (Q[:,:,-1] + P) * V - U * tau_xy - V * tau_yy

	return F

def compRHS(Q,x,y,dt,mu,gamma,step):
	nx = np.shape(x,0)
	ny = np.shape(y,1)
	lo_x = 1
	hi_x = nx-1
	lo_y = 1
	hi_y = ny-1


	# format of Q[x,y,0-3] 
	if step == 'predictor':
		E = compE(Q,x,y,mu,gamma,step)
		F = compF(Q,x,y,mu,gamma,step)

		dEdx = compute_x_deriv(E,x,y,0)
		dFdy = compute_y_deriv(F,x,y,1)

		rhs = -1 * (dEdx + dFdy)
	elif step == 'corrector':
		E = compE(Q,x,y,mu,gamma,step)
		F = compF(Q,x,y,mu,gamma,step)

		dEdx = compute_x_deriv(E,x,y,1)
		dFdy = compute_y_deriv(F,x,y,0)

		rhs = -1 * (dEdx + dFdy)
	else:
		raise Exception('Invalid step')

	return rhs
# Compute RHS for drho/dt

# def compute_rhs_rho(Rho,U,rhs_rho,dir):

# 	# only apply on interior of domain, BCs handled elsewhere

# 	if dir == 'forward':
# 		for ii in range(1,nx-1):
# 			term[ii] = -1 * (Rho[ii+1] * U[ii+1] - Rho[ii] * U[ii]) / (xg[ii+1] - xg[ii])
# 	elif dir == 'backward':
# 		for ii in range(1,nx-1):
# 			term[ii] = -1 * (Rho[ii] * U[ii] - Rho[ii-1] * U[ii-1]) / (xg[ii] - xg[ii-1])
# 	else:
# 		for ii in range(1,nx-1):
# 			term[ii] = -1 * (Rho[ii+1] * U[ii+1] - Rho[ii-1] * U[ii-1]) / (xg[ii+1] - xg[ii-1])

# 	rhs_rho[1:nx-1] += term

# 	return rhs_rho

# def compute_rhs_rhoU(Rho,U,P,rhs_rhoU,dir):

# 	term1 = np.array([nx-2,1])
# 	term2 = np.array([nx-2,1])
# 	term3 = np.array([nx-2,1])

# 	# only apply on interior of domain, BCs handled elsewhere

# 	if dir == 'forward':
# 		for ii in range(1,nx-1):
# 			term1[ii] = -1 * (Rho[ii+1] * U[ii+1]**2 - Rho[ii] * U[ii]**2) / (xg[ii+1] - xg[ii]) # rho u^2
# 			term2[ii] = -1 * (P[ii+1] - P[ii]) / (xg[ii+1] - xg[ii]) # pressure
# 			term3[ii] = 2 * mu * (U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 # d/dx(tau_xx) 
# 	elif dir == 'backward':
# 		for ii in range(1,nx-1):
# 			term1[ii] = -1 * (Rho[ii+1] * U[ii+1]**2 - Rho[ii-1] * U[ii-1]**2) / (xg[ii+1] - xg[ii-1]) # rho u^2
# 			term2[ii] = -1 * (P[ii+1] - P[ii-1]) / (xg[ii+1] - xg[ii-1]) # pressure
# 			term3[ii] = 2 * mu * (U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 # d/dx(tau_xx) 
# 	else:
# 		for ii in range(1,nx-1):
# 			term1[ii] = -1 * (Rho[ii+1] * U[ii+1]**2 - Rho[ii-1] * U[ii-1]**2) / (xg[ii+1] - xg[ii-1]) # rho u^2
# 			term2[ii] = -1 * (P[ii+1] - P[ii-1]) / (xg[ii+1] - xg[ii-1]) # pressure
# 			term3[ii] = 2 * mu * (U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 # d/dx(tau_xx) 

# 	rhs_rhoU[1:nx-1] += term1 + term2 + term3

# 	return rhs_rhoU

# def compute_rhs_Et(Et,U,P,rhs_Et):

# 	term1 = np.array([nx-2,1])
# 	term2 = np.array([nx-2,1])

# 	for ii in range(1,nx-1):
# 		term1[ii] = U[ii] * ( (Et[ii+1] - Et[ii-1]) / (xg[ii+1] - xg[ii-1]) + (P[ii+1] - P[ii-1]) / (xg[ii+1] - xg[ii-1]) ) +\
# 		(Et[ii] + P[ii]) * (U[ii+1] - U[ii-1]) / (xg[ii+1] - xg[ii-1]) # d/dx ((Et + p) * u)
# 		term2[ii] = 2 * mu * ( ((U[ii+1] - U[ii-1]) / (xg[ii+1] - xg[ii-1]))**2 +  U[ii] * \
# 		(U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 ) # d/dx (u * tau_xx)

# 	rhos_E[1:nx-1] += term1 + term2

# 	return rhos_Et
import numpy as numpy
from global import xg
from global import nx 
from global import mu
from global import gamma

# Compute RHS for drho/dt

def compute_rhs_rho(Rho,U,rhs_rho,dir):

	# only apply on interior of domain, BCs handled elsewhere

	if dir == 'forward':
		for ii in range(1,nx-1):
			term[ii] = -1 * (Rho[ii+1] * U[ii+1] - Rho[ii] * U[ii]) / (xg[ii+1] - xg[ii])
	elif dir == 'backward':
		for ii in range(1,nx-1):
			term[ii] = -1 * (Rho[ii] * U[ii] - Rho[ii-1] * U[ii-1]) / (xg[ii] - xg[ii-1])
	else:
		for ii in range(1,nx-1):
			term[ii] = -1 * (Rho[ii+1] * U[ii+1] - Rho[ii-1] * U[ii-1]) / (xg[ii+1] - xg[ii-1])

	rhs_rho[1:nx-1] += term

	return rhs_rho

def compute_rhs_rhoU(Rho,U,P,rhs_rhoU,dir):

	term1 = np.array([nx-2,1])
	term2 = np.array([nx-2,1])
	term3 = np.array([nx-2,1])

	# only apply on interior of domain, BCs handled elsewhere

	if dir == 'forward':
		for ii in range(1,nx-1):
			term1[ii] = -1 * (Rho[ii+1] * U[ii+1]**2 - Rho[ii] * U[ii]**2) / (xg[ii+1] - xg[ii]) # rho u^2
			term2[ii] = -1 * (P[ii+1] - P[ii]) / (xg[ii+1] - xg[ii]) # pressure
			term3[ii] = 2 * mu * (U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 # d/dx(tau_xx) 
	elif dir == 'backward':
		for ii in range(1,nx-1):
			term1[ii] = -1 * (Rho[ii+1] * U[ii+1]**2 - Rho[ii-1] * U[ii-1]**2) / (xg[ii+1] - xg[ii-1]) # rho u^2
			term2[ii] = -1 * (P[ii+1] - P[ii-1]) / (xg[ii+1] - xg[ii-1]) # pressure
			term3[ii] = 2 * mu * (U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 # d/dx(tau_xx) 
	else:
		for ii in range(1,nx-1):
			term1[ii] = -1 * (Rho[ii+1] * U[ii+1]**2 - Rho[ii-1] * U[ii-1]**2) / (xg[ii+1] - xg[ii-1]) # rho u^2
			term2[ii] = -1 * (P[ii+1] - P[ii-1]) / (xg[ii+1] - xg[ii-1]) # pressure
			term3[ii] = 2 * mu * (U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 # d/dx(tau_xx) 

	rhs_rhoU[1:nx-1] += term1 + term2 + term3

	return rhs_rhoU

def compute_rhs_Et(Et,U,P,rhs_Et):

	term1 = np.array([nx-2,1])
	term2 = np.array([nx-2,1])

	for ii in range(1,nx-1):
		term1[ii] = U[ii] * ( (Et[ii+1] - Et[ii-1]) / (xg[ii+1] - xg[ii-1]) + (P[ii+1] - P[ii-1]) / (xg[ii+1] - xg[ii-1]) ) +\
		(Et[ii] + P[ii]) * (U[ii+1] - U[ii-1]) / (xg[ii+1] - xg[ii-1]) # d/dx ((Et + p) * u)
		term2[ii] = 2 * mu * ( ((U[ii+1] - U[ii-1]) / (xg[ii+1] - xg[ii-1]))**2 +  U[ii] * \
		(U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 ) # d/dx (u * tau_xx)

	rhos_E[1:nx-1] += term1 + term2

	return rhos_Et

def Prim2Con(Rho,U,P):
	rhoU = Rho * U
	Et = P / (gamma - 1) + Rho / 2 * (U**2)

	return Rho, rhoU, Et

def Tauxx(U,step):

	tau_xx = np.array([nx,1])
	for ii in range(1,nx-1):
		if ii == 0:
			tau_xx[ii] = (U[ii+1] - U[ii]) / (xg[ii+1] - xg[ii])
		elif ii == nx:
			tau_xx[ii] = (U[ii] - U[ii-1]) / (xg[ii] - xg[ii-1])
		else:
			tau_[xx] = (U[ii + 1] - U[ii - 1]) / (xg[ii + 1] - xg[ii - 1])
	return tau_xx

def compE(U):

def compRHS(U,step):

	# format of U[x,y,1-4] 
	if step == 'predictor':

	elif step == 'corrector':


	else:
		raise('Invalid step')


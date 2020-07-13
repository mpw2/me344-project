import numpy as numpy
from global import xg
from global import nx 
from global import mu

# Compute RHS for drho/dt

def compute_rhs_rho(Rho,U,rhs_rho):

	# only apply on interior of domain, BCs handled elsewhere
	for ii in range(1,nx-1):
		term[ii] -1 * (Rho[ii+1] * U[ii+1] - Rho[ii-1] * U[ii-1]) / (xg[ii+1] - xg[ii-1])

	rhs_rho[1:nx-1] += term

def compute_rhs_rhoU(Rho,U,P,rhs_rhoU):

	term1 = np.array([nx-2,1])
	term2 = np.array([nx-2,1])
	term3 = np.array([nx-2,1])

	# only apply on interior of domain, BCs handled elsewhere
	for ii in range(1,nx-1):
		term1[ii] = -1 * (Rho[ii+1] * U[ii+1]**2 - Rho[ii-1] * U[ii-1]**2) / (xg[ii+1] - xg[ii-1]) # rho u^2
		term2[ii] = -1 * (P[ii+1] - P[ii-1]) / (xg[ii+1] - xg[ii-1]) # pressure
		term3[ii] = 2 * mu * (U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 # d/dx(tau_xx) 

	rhs_rhoU[1:nx-1] += term1 + term2 + term3

def compute_rhs_Et(Et,U,P,rhs_Et):

	term1 = np.array([nx-2,1])
	term2 = np.array([nx-2,1])

	for ii in range(1,nx-1):
		term1[ii] = U[ii] * ( (Et[ii+1] - Et[ii-1]) / (xg[ii+1] - xg[ii-1]) + (P[ii+1] - P[ii-1]) / (xg[ii+1] - xg[ii-1]) ) +\
		(Et[ii] + P[ii]) * (U[ii+1] - U[ii-1]) / (xg[ii+1] - xg[ii-1]) # d/dx ((Et + p) * u)
		term2[ii] = 2 * mu * ( ((U[ii+1] - U[ii-1]) / (xg[ii+1] - xg[ii-1]))**2 +  U[ii] * \
		(U[ii+1] - 2 * U[ii] + U[ii-1]) / (xg[ii+1] - xg[ii-1])**2 ) # d/dx (u * tau_xx)

	rhos_Et += term1 + term2
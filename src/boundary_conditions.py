from common import *

def apply_boundary_conditions():

	rhoInf = Pinf / (Rgas * Tinf)
	
	# apply left BC
	for jj in range(ny):
		if np.abs(yg(jj)) < jet_height:
			Q[0,jj,0] = rhoInf
			Q[0,jj,1] = rhoInf * Ujet
			Q[0,jj,2] = 0
			Q[0,jj,3] = Pinf / (gamma-1) + rhoInf / 2 * (Ujet**2)
		else:
			Q[0,jj,0] = rhoInf
			Q[0,jj,1] = 0
			Q[0,jj,2] = 0
			Q[0,jj,3] = Pinf / (gamma - 1)

	# apply top bc
	# extrapolation bc 
	Q[1:,ny,:] = 2 * Q[1:,ny-1,:] - Q[1:,ny-2,:]

	# apply bottom bc
	# extrapolation bc
	Q[1:,0,:] = 2 * Q[1:,2,:] - Q[1:,3,:]

    
	# apply right bc (do not include jmin or jmax)
	# extrapolation BC
	Q[nx,1:ny-1,:] = 2 * Q[nx-1,1:ny-1,:] - Q[nx-2,1:ny-1,:]
























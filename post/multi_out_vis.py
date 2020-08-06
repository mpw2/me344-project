import os
import pickle
import sys
import pdb

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def main():
    
    if len(sys.argv) == 1:
        raise Exception('Usage: multi_out_vis.py [nprocs]')

    nprocs = int(sys.argv[1])

    gamma = 1.4

    files = sorted(os.listdir("./out/"))
    numFiles = len(files)

    # this loop is for loading the grid. only needs to be done for one time step
    for ii in range(nprocs):
        
        dataFile = files[ii]
        rank = int(dataFile.split(".")[2])

        with open('./out/' + dataFile, 'rb') as f:
            saveVars = pickle.load(f)

        if rank % nprocs == 0:
            x = saveVars[0][0:-1,:,:]
            y = saveVars[1][:,:,:]
            z = saveVars[2][:,:,:]

        elif rank % nprocs != nprocs - 1:
            x = np.concatenate((x,saveVars[0][1:-1,:,:]),axis=0)
            #y = np.concatenate((y,saveVars[1][:,1:-1,:]),axis=1)
            #z = np.concatenate((z,saveVars[2][:,:,1:-1]),axis=2)

        else:
            x = np.concatenate((x,saveVars[0][1:,:,:]),axis=0)
            #y = np.concatenate((y,saveVars[1][:,1:,:]),axis=1)
            #z = np.concatenate((z,saveVars[2][:,:,1:]),axis=2)

    nx = x.size
    ny = y.size
    nz = z.size

    x = np.tile(x, (1, ny, nz))
    y = np.tile(y, (nx, 1, nz))
    z = np.tile(z, (nx, ny, 1))

    for dataFile in sorted(os.listdir("./out/")):

        step = dataFile.split(".")[1]
        rank = int(dataFile.split(".")[2])

        with open('./out/' + dataFile, 'rb') as f:
            saveVars = pickle.load(f)

        if rank % nprocs == 0:
            Q = saveVars[3][0:-1,:,:,:]

        elif rank % nprocs != nprocs - 1:
            Q = np.concatenate((Q,saveVars[3][1:-1,:,:,:]), axis = 0)

        else:
            Q = np.concatenate((Q,saveVars[3][1:,:,:,:]),axis=0)

            Rho = Q[:,:,:,0]
            U = Q[:,:,:,1] / Q[:,:,:,0]
            V = Q[:,:,:,2] / Q[:,:,:,0]
            W = Q[:,:,:,3] / Q[:,:,:,0]
            P = (gamma - 1) * (Q[:,:,:,4] - 0.5 / Q[:,:,:,0] * (Q[:,:,:,1] + Q[:,:,:,2] + Q[:,:,:,3])**2)
            Phi = Q[:,:,:,5] / Q[:,:,:,0]

            plot_midplane_contour(x,y,nz,Rho,r'$\rho$','images/rho/rho' + '.' + step + '.png')
            plot_midplane_contour(x,y,nz,U,r'$u$','images/u/u' + '.' + step + '.png')
            plot_midplane_contour(x,y,nz,V,r'$v$','images/v/v' + '.' + step + '.png')
            plot_midplane_contour(x,y,nz,W,r'$w$','images/w/w' + '.' + step + '.png')
            plot_midplane_contour(x,y,nz,P,r'$P$','images/p/p' + '.' + step + '.png')
            plot_midplane_contour(x,y,nz,Phi,r'$\phi$','images/phi/phi' + '.' + step + '.png')

            # plot isosurface

            plot_isocontour(x,y,z,Phi,0.1,0.8,3,r'$\phi$','isocontours/phi/phi' + '.' + step + '.png')

def plot_isocontour(X,Y,Z,values,minVal,maxVal,numSurfaces,titleArg,savePath):
    fig = go.Figure(data=go.Isosurface(
        x = X.flatten(),
        y = Y.flatten(),
        z = Z.flatten(),
        value=values.flatten(),
        isomin=minVal,
        isomax=maxVal,
        surface_count=numSurfaces))

    #fig.update_layout(
    #    title=titleArg
    #    xaxis_title='x'
    #    yaxis_title='y'
    #    zaxis_title='z')

    fig.write_image(savePath)

def plot_midplane_contour(x,y,nz,var,titleArg,savePath):

    plt.figure()
    plt.contourf(x[:, :, 0], y[:, :, 0], var[:, :, int(nz/2)], 100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titleArg)
    plt.colorbar()
    plt.savefig(savePath)
    plt.close('all')

if __name__ == "__main__":
    main()

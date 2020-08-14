import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def main():

    for dataFile in sorted(os.listdir("./out/")):

        # split data file to get step number
        step = dataFile.split(".")[1]

        with open('./out/' + dataFile, 'rb') as f:
            saveVars = pickle.load(f)

        x = saveVars[0]
        y = saveVars[1]
        z = saveVars[2]
        Q = saveVars[3]

        # need to convert x and y to 2d arrays
        nx = x.size
        ny = y.size
        nz = z.size

        x = np.tile(x, (1, ny, nz))
        y = np.tile(y, (nx, 1, nz))
        z = np.tile(z, (nx, ny, 1))

        gamma = 1.4

        Rho = Q[:, :, :, 0]
        U = Q[:, :, :, 1] / Q[:, :, :, 0]
        V = Q[:, :, :, 2] / Q[:, :, :, 0]
        W = Q[:, :, :, 3] / Q[:, :, :, 0]
        P = (gamma - 1) * (Q[:, :, :, 4] - 0.5 / Q[:, :, :, 0] *
                           (Q[:, :, :, 1] + Q[:, :, :, 2] +
                            Q[:, :, :, 3])**2)
        Phi = Q[:, :, :, 5] / Q[:, :, :, 0]

        plot_midplane_contour(x,y,nz,Rho,r'$\rho$','images/rho/rho' + '.' + step + '.png')
        plot_midplane_contour(x,y,nz,U,r'$u$','images/u/u' + '.' + step + '.png')
        plot_midplane_contour(x,y,nz,V,r'$v$','images/v/v' + '.' + step + '.png')
        plot_midplane_contour(x,y,nz,W,r'$w$','images/w/w' + '.' + step + '.png')
        plot_midplane_contour(x,y,nz,P,r'$P$','images/p/p' + '.' + step + '.png')
        plot_midplane_contour(x,y,nz,Phi,r'$\phi$','images/phi/phi' + '.' + step + '.png')

        # plot isosurface

        plot_isocontour(x,y,z,Phi,0.1,0.8,3,r'$\phi$','isocontours/phi/phi' + '.' + step + '.png')

        print('Saved step ' + step + ' images')
        print('')


def plot_isocontour(X,Y,Z,values,minVal,maxVal,numSurfaces,titleArg,savePath):
    fig = go.Figure(data=go.Isosurface(
    x = X.flatten(),
    y = Y.flatten(),
    z = Z.flatten(),
    value = values.flatten(),
    isomin = minVal,
    isomax = maxVal,
    surface_count = numSurfaces))

    fig.write_image(savePath)

def plot_midplane_contour(x,y,nz,var,titleArg,savePath):
    plt.figure()
    plt.contourf(x[:,:,0],y[:,:,0],var[:,:,int(nz/2)],100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titleArg)
    plt.colorbar()
    plt.savefig(savePath)
    plt.close('all')

if __name__ == "__main__":
    main()


#

#!/usr/bin/env python

import os
import pickle
import sys
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt


def plot_file(dataPath):
        dataDir, dataFile = os.path.split(dataPath) 
        try:
            step = dataFile.split(".")[1]
        except:
            return

        with open(dataPath, 'rb') as f:
            saveVars = pickle.load(f, encoding='latin1')

        # saveVars = [x, y, z, Q]
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

        # Plot and save figures
        plt.figure()
        plt.contourf(x[:, :, 0], y[:, :, 0], Rho[:, :, int(nz/2)], 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$\rho$')
        plt.colorbar()
        plt.savefig('images/rho/rho' + '.' + step + '.png')

        plt.figure()
        plt.contourf(x[:, :, 0], y[:, :, 0], U[:, :, int(nz/2)], 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$U$')
        plt.colorbar()
        plt.savefig('images/u/u' + '.' + step + '.png')

        plt.figure()
        plt.contourf(x[:, :, 0], y[:, :, 0], V[:, :, int(nz/2)], 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$V$')
        plt.colorbar()
        plt.savefig('images/v/v' + '.' + step + '.png')

        plt.figure()
        plt.contourf(x[:, :, 0], y[:, :, 0], W[:, :, int(nz/2)], 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$W$')
        plt.colorbar()
        plt.savefig('images/w/w' + '.' + step + '.png')

        plt.figure()
        plt.contourf(x[:, :, 0], y[:, :, 0], P[:, :, int(nz/2)], 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$P$')
        plt.colorbar()
        plt.savefig('images/p/p' + '.' + step + '.png')

        plt.figure()
        plt.contourf(x[:, :, 0], y[:, :, 0], Phi[:, :, int(nz/2)], 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$\phi$')
        plt.colorbar()
        plt.savefig('images/phi/phi' + '.' + step + '.png')

        plt.close('all')

        print('Saved step ' + step + ' images')


def main():
    
    if len(sys.argv) == 2:
        dataFile = sys.argv[1]
        plot_data(dataFile)

    else:
        data_dir = "./out/"
        dataFiles = os.listdir(data_dir)
        dataPaths = [data_dir + df for df in dataFiles]
        pool = mp.Pool(processes=8)
        pool.map(plot_file, dataPaths)
        pool.close()
        pool.join()
        print('Done!')


if __name__ == "__main__":
    main()


#

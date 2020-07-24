#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os


def main():

    if len(sys.argv) == 2:

        dataFile = sys.argv[1]

        # f = open(dataFile, 'rb')
        # saveVars = np.load(f)

        # print(type(saveVars))

        with open(dataFile, 'rb') as f:
            saveVars = pickle.load(f)

        # saveVars = [x, y, Q]

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

        Rho = Q[:, :, :, 0]
        U = Q[:, :, :, 1] / Q[:, :, :, 0]
        V = Q[:, :, :, 2] / Q[:, :, :, 0]
        W = Q[:, :, :, 3] / Q[:, :, :, 0]

        # print(Q)
        # Rho, U, V, P = eq.ConsToPrim(Q)

        plt.figure()
        plt.contourf(x, y, Rho, 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$\rho$')
        plt.colorbar()
        plt.savefig('images/rho.png')

        plt.figure()
        plt.contourf(x, y, U, 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$U$')
        plt.colorbar()
        plt.savefig('images/u.png')

        plt.figure()
        plt.contourf(x, y, V, 100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$V$')
        plt.colorbar()
        plt.savefig('images/v.png')

    else:

        for dataFile in os.listdir("./out/"):

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

            # print(Q)
            # Rho, U, V, P = eq.ConsToPrim(Q)

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
            print('')


if __name__ == "__main__":
    main()


#

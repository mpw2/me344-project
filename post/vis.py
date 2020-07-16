import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os

def main():

    if len(sys.argv) == 2:

        dataFile = sys.argv[1]

        # f = open(dataFile,'rb')
        # saveVars = np.load(f)

        # print(type(saveVars))

        with open(dataFile,'rb') as f:
            saveVars = pickle.load(f)

        # saveVars = [x, y, Q]

        x = saveVars[0]
        y = saveVars[1]
        Q = saveVars[2]

        # need to convert x and y to 2d arrays
        nx = x.size
        ny = y.size

        x = np.tile(x,(1,ny))
        y = np.tile(y,(nx,1))

        Rho = Q[:,:,0]
        U = Q[:,:,1] / Q[:,:,0]
        V = Q[:,:,2] / Q[:,:,0]

        #print(Q)
        #Rho, U, V, P = eq.ConsToPrim(Q)


        fig = plt.figure()
        plt.contourf(x,y,Rho,100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$\rho$')
        plt.colorbar()
        plt.savefig('images/rho.png')

        fig = plt.figure()
        plt.contourf(x,y,U,100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$U$')
        plt.colorbar()
        plt.savefig('images/u.png')

        fig = plt.figure()
        plt.contourf(x,y,V,100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$V$')
        plt.colorbar()
        plt.savefig('images/v.png')

    else:

        for dataFile in os.listdir("./out/"):

            # split data file to get step number
            step = dataFile.split(".")[1]

            with open('./out/' + dataFile,'rb') as f:
                saveVars = pickle.load(f)

            x = saveVars[0]
            y = saveVars[1]
            Q = saveVars[2]

            # need to convert x and y to 2d arrays
            nx = x.size
            ny = y.size

            x = np.tile(x,(1,ny))
            y = np.tile(y,(nx,1))

            Rho = Q[:,:,0]
            U = Q[:,:,1] / Q[:,:,0]
            V = Q[:,:,2] / Q[:,:,0]

            #print(Q)
            #Rho, U, V, P = eq.ConsToPrim(Q)


            fig = plt.figure()
            plt.contourf(x,y,Rho,100)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(r'$\rho$')
            plt.colorbar()
            plt.savefig('images/rho/rho' + '.' + step + '.png')

            fig = plt.figure()
            plt.contourf(x,y,U,100)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(r'$U$')
            plt.colorbar()
            plt.savefig('images/u/u' + '.' + step + '.png')

            fig = plt.figure()
            plt.contourf(x,y,V,100)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(r'$V$')
            plt.colorbar()
            plt.savefig('images/v/v' + '.' + step + '.png')

            plt.close('all')



if __name__ == "__main__":
    main()
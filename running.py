import initialization
import numpy as np
import scipy.sparse as sparse
import time
import computations

massDict = {'H': 1.0080, 'O': 15.9994, 'C': 12.0110} #From character to amu = g/mol

def remapping_coords(L):
    """Remaps the center of mass of the atoms back into the L x L x L box
    
    INPUT:
    L : box size
    
    UPDATES:
    currentCoord : global variable with xyz coordinates of each atom
    """

    #update global variable to minimize copying of currentCoord
    global currentCoord

    cmWater = waterMatrix@currentCoord[:nr_water*3]
    cmEthanol = ethanolMatrix@currentCoord[nr_water*3:]
    centerMasses = np.concatenate((cmWater, cmEthanol))

    centerMasses = (np.mod(centerMasses, L) - centerMasses)
    centerMassesNew = np.repeat(centerMasses, [3]*nr_water + [9]*nr_ethanol, axis=0)
    currentCoord += centerMassesNew #currentCoord is a global variable

def simulateSystem(AtomInfo: np.array, coordinates: np.array, dt : float, nr_timesteps : int, temperature: float, moleculesVelocity : np.array, L = 30, r = 20):
    """Simulates the sytem with molecules having coordinates moleculesCoords, and atoms as given in moleculesInfo.

    REQUIRED BEFOREHAND:
    nr_water : nr of water molecules
    nr_ethanol : nr of ethanol molecules
    bond, angle and dihedral information : for the different forces
    waterMatrix and ethanolMatrix : matrices used to compute center of mass
    measure_rows_O, measure_cols_O,  measure_cols_H
    currentCoord : xyz coordinates  of all atoms (for global use)
    

    INPUT:
    AtomInfo : ["O", "H"] list for all atoms
    coordinates : initial xyz coordinates of all atoms
    dt : timestep
    nr_timesteps : length of simulation
    temperature : desired temperature system
    moleculesVelocity : intial xyz velocity of all atoms
    L : box size
    r : cutoff radius

    OUTPUT:
    WaterSimulateSystem.xyz : txt file with xyz coordinates water
    EthanolSimulateSystem.xyz : txt file with xyz coordinates ethanol 
    """

    fileNameWater = open("./WaterSimulateSystem.xyz", "w+")
    fileNameEthanol = open("./EthanolSimulateSystem.xyz","w+")
    filenameTemperatureKineticBefore = open("./TemperatureKineticBefore.txt","w+")
    filenameTemperatureKineticAfter = open("./TemperatureKineticAfter.txt","w+")
    filenameEnergy = open("./Energy.txt", "w+")
    nrAtoms = len(AtomInfo)

    atom_masses = np.array([massDict[atom] for atom in AtomInfo])
    
    global currentCoord
    currentCoord = coordinates
    currentVelocity = moleculesVelocity
    
    for iteration in range(nr_timesteps):    
        t0 =  time.time()   
        print('this is iteration n = ', iteration)
        fileNameWater.write(str(nr_water*3) + '\n')
        fileNameWater.write("t=" + str(iteration*dt) + ' \n')
        fileNameEthanol.write(str(nr_ethanol*9) + '\n')  
        fileNameEthanol.write("t=" + str(iteration*dt) + ' \n')

        filenameTemperatureKineticBefore.write("t=" + str(iteration*dt) + ' \n')
        filenameTemperatureKineticAfter.write("t=" + str(iteration*dt) + ' \n')
        filenameEnergy.write("t=" + str(iteration*dt) + ' \n')

        if iteration == 0: #starting iteration
            beforeTemp, beforeKin = computations.thermometer(currentVelocity, atom_masses, nrAtoms)
            filenameTemperatureKineticBefore.write(str(beforeTemp) + ' \t' + str(beforeKin)  + ' \n')
            
            #rescale velocities for proper starting temperature
            currentVelocity *= computations.thermostat(currentVelocity, atom_masses, nrAtoms, temperature)
            computedTemp, kinEnergy = computations.thermometer(currentVelocity, atom_masses, nrAtoms)
            filenameTemperatureKineticAfter.write(str(computedTemp) + ' \t' + str(kinEnergy)  + ' \n')

            #calculate forces current position
            atomforce, energyLJ, energyBond, energyAngle, energyDihedral  = computations.calc_total_force(nrAtoms, nr_water, currentCoord, bond_indices, bond_constants, angle_indices, angle_constants, dihedral_indices, dihedral_constants, L, r, sigma, epsilon, blockMatrix_neighbours)
            energyMatrix[iteration, :] = [energyLJ, energyBond, energyAngle, energyDihedral]

            #update coordinates and compute new forces
            currentCoord += dt*currentVelocity + (dt**2/2)*(1/atom_masses)[:, np.newaxis]*atomforce

            # Map coordinates back into the box, based on position center of mass
            remapping_coords(L)

            #compute forces new position
            atomforce_new, energyLJ, energyBond, energyAngle, energyDihedral  = computations.calc_total_force(nrAtoms, nr_water, currentCoord, bond_indices, bond_constants, angle_indices, angle_constants, dihedral_indices, dihedral_constants, L, r, sigma, epsilon, blockMatrix_neighbours)
            atomforce = atomforce_new


        else:
            #Velocity Verlet integration
            #update coordinates
            currentCoord += dt*currentVelocity + (dt**2/2)*(1/atom_masses)[:, np.newaxis]*atomforce

            # Map coordinates back into the box, based on position center of mass
            remapping_coords(L)
            energyMatrix[iteration, :] = [energyLJ, energyBond, energyAngle, energyDihedral]

            #compute new forces new position
            atomforce_new, energyLJ, energyBond, energyAngle, energyDihedral = computations.calc_total_force(nrAtoms, nr_water, currentCoord, bond_indices, bond_constants, angle_indices, angle_constants, dihedral_indices, dihedral_constants, L, r, sigma, epsilon, blockMatrix_neighbours)

            #updpate velocity
            currentVelocity += (dt/2)*(1/atom_masses)[:, np.newaxis]*(atomforce_new + atomforce)

            beforeTemp, beforeKin = computations.thermometer(currentVelocity, atom_masses, nrAtoms)
            filenameTemperatureKineticBefore.write(str(beforeTemp) + ' \t' + str(beforeKin) + ' \n')
            
            currentVelocity *= computations.thermostat(currentVelocity, atom_masses, nrAtoms, temperature)
            computedTemp, kinEnergy = computations.thermometer(currentVelocity, atom_masses, nrAtoms)
            filenameTemperatureKineticAfter.write(str(computedTemp) + ' \t' + str(kinEnergy) + ' \n')

            atomforce = atomforce_new

        for atom in range(nr_water*3):
            fileNameWater.write( AtomInfo[atom] + '\t' + str(currentCoord[atom][0]) + '\t' + str(currentCoord[atom][1]) + '\t' + str(currentCoord[atom][2]) + '\n' )
        for atom in range(nr_water*3, nr_water*3 + nr_ethanol*9):
            fileNameEthanol.write( AtomInfo[atom] + '\t' + str(currentCoord[atom][0]) + '\t' + str(currentCoord[atom][1]) + '\t' + str(currentCoord[atom][2]) + '\n' )

        print('the iteration took ', time.time() - t0, 's')

    np.savetxt("energyResults.csv", energyMatrix, delimiter=',')
    fileNameWater.close()
    fileNameEthanol.close()
    filenameTemperatureKineticBefore.close()
    filenameTemperatureKineticAfter.close()
    filenameEnergy.close()

nr_water = int(0)
nr_ethanol = int(277)
waterList = np.append( np.ones(nr_water, dtype = 'bool') , np.zeros(nr_ethanol, dtype = 'bool'))

#initialize matrices to compute the center of mass
waterMatrix, ethanolMatrix = initialization.cm_matrix(nr_water, nr_ethanol)

#generate molecules
atoms_info, currentCoord, bond_indices, bond_constants, angle_indices, angle_constants, dihedral_indices, dihedral_constants = initialization.generate_molecules(nr_water, nr_ethanol)

initialVel = initialization.initializeVelocity(0.1,3*nr_water + 9*nr_ethanol)
#save the sigma, epsilon en neighbours matrix used for the LJ force
initialization.intializeSigmaEps(waterList)

#once correct sigma, epsilon and neighbours matrix has been saved the first time: only need to load
blockMatrix_neighbours = np.load("blockMatrix_neighbours.npy", allow_pickle= True)
sigma = sparse.load_npz("sigma_sparse.npz")
epsilon = sparse.load_npz("epsilon_sparse.npz")

timeStep = 0.02 #1 in simulation = 0.1 ps, so 2 femtoseconds
tBegin = 0
numberTimes = int((1/2)*10**5)
temp = 298.15 #Kelvin

energyMatrix = np.empty([numberTimes, 4])

print('starting simulation, done initializing')
tbegin = time.time()
simulateSystem(atoms_info, currentCoord, timeStep, numberTimes, temp, initialVel)
print('full simulation done! took in total ', time.time() - tbegin, ' s')


import numpy as np
import scipy.sparse as sparse
import scipy

def readXYZfile(fileName):
    """Read a .xyz file.

    INPUT: .xyz file
    OUTPUT: atom type, xyz-coord for every atom.
    """

    #Opening the file
    with open(fileName, "r") as file:
      nr_atoms = int(file.readline())
      array = np.loadtxt(fileName, skiprows=2, max_rows=nr_atoms, dtype = str)
      
      atoms = array[:, 0]
      coordinates = array[:, 1:].astype("float32")
      file.close()

    return atoms, coordinates

def readTopology(filename):
      """Reads a topology file

       INPUT: topology file
       OUTPUT: bond, angles and dihedral information
       """

      with open(filename, "r") as file:
        nr_of_bonds = int(np.loadtxt(filename, max_rows=1, dtype = str)[1])
        bonds_array = np.loadtxt(filename, skiprows=1, max_rows=nr_of_bonds, dtype = str)

        nr_of_angles = int(np.loadtxt(filename, skiprows = (1 + nr_of_bonds), max_rows=1, dtype = str)[1])
        angles_array = np.loadtxt(filename, skiprows = 2 + nr_of_bonds, max_rows=nr_of_angles, dtype = str)

        nr_of_dihedrals = int(np.loadtxt(filename, skiprows = (2 + nr_of_bonds + nr_of_angles), max_rows=1, dtype = str)[1])
        if nr_of_dihedrals:
          dihedrals_array = np.loadtxt(filename, skiprows = 3 + nr_of_bonds + nr_of_angles, max_rows=nr_of_dihedrals, dtype = str)
        else:
          dihedrals_array = None

      file.close()

      return bonds_array, angles_array, dihedrals_array

def generate_molecules(nrWater: int, nrEthanol: int, boxsize = 30, dist = 2):
    """Generates nrWater and nrEthanol molecules on a grid in a box

    INPUT:
        nrWater : number of water molecules to generate
        nrEthanol : number of ethanol molecules to generate
        boxsize : dimensions of box
        dist : minimum distance needed between molecules

    OUTPUT:
        atom_info : e.g. ["O", "H", ..]
        coordinates : xyz coordinates for each atom
        bond_indices : list with e.g. [[1 2], .. ] for a bond between atoms 1 - 2
        bond_constants: [R, K_b] for each bond
        angle_indices: list with e.g. [[1 2 3], .. ] for an angle between atoms 1 - 2 - 3
        angle_constants: [theta, k_theta] for each angle
        dihedral_indices: list with e.g. [1 2 3 4] for a dihedral between atoms 1 - 2 - 3 - 4
        dihedral_constants: [C_1 C_2 C_3 C_4] for each dihedral
    """

    n = int(boxsize/(2*dist)) #nr of boxes to place ethanol molecules

    availableBoxes = [ [dist + i*2*dist,  dist + j*2*dist , dist + k*2*dist] for i in range(n) for j in range(n) for k in range(n)]
    availableBoxesOld = [ [dist + i*2*dist,  dist + j*2*dist , dist + k*2*dist] for i in range(n) for j in range(n) for k in range(n)]
    indicesAvailable = [i for i in range(len(availableBoxes))]
    indCentersEthanol = np.random.choice(indicesAvailable, size = nrEthanol, replace = False)
    for item in indCentersEthanol:
        availableBoxes.remove(availableBoxesOld[item])

    availableWater = []
    for box in availableBoxes:
        for i in np.arange(start = -1, stop = 3, step = 2):
            for j in np.arange(start = -1, stop = 3, step = 2):
                for k in np.arange(start = -1, stop = 3, step = 2):
                    availableWater.append( [ box[0] + i*(1/2)*dist, box[1] +j*(1/2)*dist, box[2] + k*(1/2)*dist ])

    indicesAvailableWater = [i for i in range(len(availableWater))]
    indCentersWater = np.random.choice(indicesAvailableWater, size = nrWater, replace = False)
    centersWater = [availableWater[i] for i in indCentersWater]
    centersEthanol = [availableBoxesOld[i] for i in indCentersEthanol]

    #extend and shift bond_info indices
    if nrWater > 0:
        water_atoms, water_base_pos = readXYZfile("water.xyz")
        water_bond_info, water_angle_info, water_dihedral = readTopology("water_constants.txt")
        additions_water_bonds = 3*np.repeat(range(nrWater), len(water_bond_info))
        water_bond_info_indices = np.tile(water_bond_info[:, 0:2].astype('int'), [nrWater, 1]) + additions_water_bonds[:, np.newaxis]
        water_bond_info = np.tile(water_bond_info[:, 2:].astype('float32'), [nrWater, 1])

        additions_water_angles = 3*np.repeat(range(nrWater), len(water_angle_info[0]))
        water_angle_info_indices = np.tile(water_angle_info[0:3].astype('int'), [nrWater, 1]) + additions_water_angles[:, np.newaxis]
        water_angle_info = np.tile(water_angle_info[3:].astype('float32'), [nrWater, 1])

        water_coords = np.tile(water_base_pos, [nrWater, 1])
        water_coords += np.repeat(centersWater, nrWater*[len(water_atoms)], axis = 0)
        water_atoms = np.tile(water_atoms, nrWater)

    if nrEthanol > 0:
        ethanol_atoms, ethanol_base_pos = readXYZfile("Ethanol.xyz")
        ethanol_bond_info, ethanol_angle_info, ethanol_dihedral_info = readTopology("Ethanol_constants.txt")
        additions_ethanol_bonds = nrWater*3 + 9*np.repeat(range(nrEthanol), len(ethanol_bond_info))
        ethanol_bond_info_indices = np.tile(ethanol_bond_info[:, 0:2].astype('int'), [nrEthanol, 1]) + additions_ethanol_bonds[:, np.newaxis]
        ethanol_bond_info = np.tile(ethanol_bond_info[:, 2:].astype('float32'), [nrEthanol, 1])

        additions_ethanol_angles = nrWater*3 + 9*np.repeat(range(nrEthanol), len(ethanol_angle_info))
        ethanol_angle_info_indices = np.tile(ethanol_angle_info[:, 0:3].astype('int'), [nrEthanol, 1]) + additions_ethanol_angles[:, np.newaxis]
        ethanol_angle_info = np.tile(ethanol_angle_info[:, 3:].astype('float32'), [nrEthanol, 1])

        additions_ethanol_dihedrals = nrWater*3 + 9*np.repeat(range(nrEthanol), len(ethanol_dihedral_info))
        ethanol_dihedral_info_indices = np.tile(ethanol_dihedral_info[:, 0:4].astype('int'), [nrEthanol, 1]) + additions_ethanol_dihedrals[:, np.newaxis]
        ethanol_dihedral_info = np.tile(ethanol_dihedral_info[:, 4:].astype('float32'), [nrEthanol, 1])

        ethanol_coords = np.tile(ethanol_base_pos, [nrEthanol, 1])
        ethanol_coords += np.repeat(centersEthanol, nrEthanol*[len(ethanol_atoms)], axis = 0)
        ethanol_atoms = np.tile(ethanol_atoms, nrEthanol)
    
    if nrEthanol > 0 and nrWater > 0: 
        bond_indices = np.concatenate((water_bond_info_indices, ethanol_bond_info_indices))
        bond_constants = np.concatenate((water_bond_info, ethanol_bond_info), dtype = 'float32')
        angle_indices = np.concatenate((water_angle_info_indices, ethanol_angle_info_indices))
        angle_constants = np.concatenate((water_angle_info, ethanol_angle_info))
        dihedral_indices = ethanol_dihedral_info_indices
        dihedral_constants = ethanol_dihedral_info
        atoms_info = np.concatenate((water_atoms, ethanol_atoms), axis = 0)
        coordinates = np.concatenate((water_coords, ethanol_coords), axis = 0)
        return atoms_info, coordinates, bond_indices, bond_constants, angle_indices, angle_constants, dihedral_indices, dihedral_constants
    elif nrEthanol > 0:
        return ethanol_atoms, ethanol_coords, ethanol_bond_info_indices, ethanol_bond_info, ethanol_angle_info_indices, ethanol_angle_info, ethanol_dihedral_info_indices, ethanol_dihedral_info
    elif nrWater > 0:
        return water_atoms, water_coords, water_bond_info_indices, water_bond_info, water_angle_info_indices, water_angle_info, [0], [0]

def intializeSigmaEps(water_list):
    """Compute sigma and epsilon matrix for LJ force
    sigma and epsilon are 0 for atoms from same molecule.

    INPUT:
    water_list : 1 for water atom, 0 for ethanol atom

    SAVES FILES:
    sigma : sparse triu matrix with (sigma_i + sigma_j)/2 for atoms i, j
    epsilon : sparse triu matrix with sqrt(eps_i * e_j) for atoms i, j
    neighbours : entry [i, j] = 1 if atoms i, j in same molecule
    """

    water_atoms = ["O", "H", "H"]
    ethanol_atoms = ["C", "H_C", "H_C", "H_C", "C", "H_C", "H_C", "O", "H_O"]
    trans_dict_water = {"O": [0.315061*10, 0.66386], "H": [0, 0]} #sigma, eps
    trans_dict_ethanol = {"H_C": [2.5, 0.125520], "H_O": [0, 0], "C": [3.5, 0.276144], "O": [3.12, 0.711280]} #sigma, eps

    LJ_info_water = np.array([trans_dict_water[atom] for atom in water_atoms], ndmin=2)
    sigma_water_water = np.asarray([[0.315061*10, (1/2)*0.315061*10, (1/2)*0.315061*10], [(1/2)*0.315061*10, 0, 0], [(1/2)*0.315061*10, 0, 0]])
    eps_water_water = np.asarray([[0.66386, 0, 0], [0, 0, 0], [0, 0, 0]])

    LJ_info_eth = np.array([trans_dict_ethanol[atom] for atom in ethanol_atoms], ndmin=2)
    sigma_eth_eth = 1/2*(LJ_info_eth[:, 0] + (LJ_info_eth[:, 0])[:, np.newaxis])
    eps_eth_eth = np.sqrt(LJ_info_eth[:, 1][:, np.newaxis]*LJ_info_eth[:, 1])

    sigma_eth_water = 1/2*(LJ_info_water[:, 0] + (LJ_info_eth[:, 0])[:, np.newaxis])
    eps_eth_water = np.sqrt(LJ_info_eth[:, 1][:, np.newaxis]*LJ_info_water[:, 1])

    eth_matrix_dict = {0: [sigma_eth_eth, eps_eth_eth], 1: [sigma_eth_water, eps_eth_water]} #0 - 0 and 0 - 1
    water_matrix_dict = {0: [sigma_eth_water.T, eps_eth_water.T], 1: [sigma_water_water, eps_water_water]} #1 - 0 and 1 - 1

    nr_of_molec = (np.sum(water_list)*3) + (np.sum(1 - water_list)*9)

    sigmas = np.zeros((nr_of_molec, nr_of_molec))
    eps = np.zeros((nr_of_molec, nr_of_molec))

    index = 0
    for k in range(len(water_list)):
        if water_list[k] == 0: #ethanol
            sigma_row = np.block([eth_matrix_dict[j][0] for j in water_list])
            eps_row = np.block([eth_matrix_dict[j][1] for j in water_list])
        else: #water
            sigma_row = np.block([water_matrix_dict[j][0] for j in water_list])
            eps_row = np.block([water_matrix_dict[j][1] for j in water_list])

        sigmas[index: (index + 3*water_list[k] + 9*(1 - water_list[k])), :] = sigma_row
        eps[index:(index + 3*water_list[k] + 9*(1 - water_list[k])), :] = eps_row
        index += 3*water_list[k] + 9*(1 - water_list[k])

    neighbours = scipy.linalg.block_diag(*[np.ones([3*i + 9*(1 - i), 3*i + 9*(1 - i)]) for i in water_list])
    neighbours = neighbours.astype('bool')
    sigmas = sparse.triu(sigmas, format = "csr").multiply(1 - neighbours)
    eps = sparse.triu(eps, format = "csr").multiply(1 - neighbours)

    sparse.save_npz("sigma_sparse.npz", sigmas)
    sparse.save_npz("epsilon_sparse.npz", eps)
    np.save("blockMatrix_neighbours.npy", neighbours)

def initializeVelocity(sd : float, nrOfAtoms : int):
    """ Returns randomized initial velocities
    INPUT:
        sd : standard deviation
        nrOfAtoms
        
    OUTPUT:
        xyz vector for each atom
    """

    return( np.random.normal(0.0, sd, (nrOfAtoms,3)) )


def cm_matrix(nr_water, nr_ethanol):
    """Creates matrices used to compute center of mass

    INPUT:
    nr_water : nr of water molecules
    nr_ethanol : nr of ethanol molecules
        
    OUTPUT:
    waterMatrix : matrix to compute center of mass of water molecules
    ethanolMatrix : matrix to compute center of mass of ethanol molecules 
    """

    water_mass = np.array([15.9994, 1.0080, 1.0080])
    total_mass_water = np.sum(water_mass)
    ethanol_mass = np.array([12.0110, 1.0080, 1.0080, 1.0080, 12.0110, 1.0080, 1.0080,  15.9994, 1.0080])
    total_ethanol_mass = np.sum(ethanol_mass)

    waterMatrix = np.zeros((nr_water, nr_water*3), dtype = "float32")

    for i in range(nr_water):
        waterMatrix[i, i*3 : (i+1)*3] = water_mass/total_mass_water
    ethanolMatrix = np.zeros((nr_ethanol, nr_ethanol*9), dtype = "float32")

    for i in range(nr_ethanol):
        ethanolMatrix[i, i*9 : (i+1)*9] = ethanol_mass/total_ethanol_mass

    return waterMatrix, ethanolMatrix
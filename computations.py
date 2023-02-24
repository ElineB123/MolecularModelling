import numpy as np
import scipy.sparse as sparse
import time

def calc_bond_force(coordinates: np.array, bond_indices: np.array, bond_constants: np.array):
    """Computes total bond energy and force for all bonds in bond info. 

    INPUT:
    coordinates : xyz for each atom
    bond_indices : list with e.g. [[1 2], .. ] for a bond between atoms 1 - 2
    bond_constants: [R, K_b] for each bond

    OUTPUT :
    energyBond : total potential bond energy of all atoms
    force : total bond force on all atoms
    """

    elts1 = bond_indices[:, 0]
    elts2 = bond_indices[:, 1]

    r_ij = coordinates[elts2, :] - coordinates[elts1, :]
    norm_r = np.linalg.norm(r_ij, ord = 2, axis=1)
    
    k_bonds = bond_constants[:, 1]
    r_bonds = bond_constants[:, 0]

    factBond = k_bonds*(norm_r - r_bonds)

    bond_force = (factBond/norm_r)[:, np.newaxis]*r_ij
    energyBond = np.sum((factBond**2)/(2*k_bonds))

    force = np.zeros((len(coordinates), 3))
    np.add.at(force, elts1, bond_force)
    np.add.at(force, elts2, -bond_force)

    return energyBond, force

def calc_angular_force(coords: np.array, angle_indices: np.array, angle_constants: np.array):
       """Computes angular energy and force for all angles in angle info. 

       INPUT:
       coordinates : xyz for each atom
       angle_indices: list with e.g. [[1 2 3], .. ] for an angle between atoms 1 - 2 - 3
       angle_constants: [theta, k_theta] for each angle

       OUTPUT :
       energyAngular : total angular energy from angular potential
       force : total angular force on all atoms
       """

       elts1 = angle_indices[:, 0]
       elts2 = angle_indices[:, 1]
       elts3 = angle_indices[:, 2]

       coords_a = coords[elts1]
       coords_b = coords[elts2]
       coords_c = coords[elts3]

       r_ab = coords_b - coords_a
       r_bc = coords_c - coords_b
       r_ac = coords_c - coords_a

       norm_ab = np.linalg.norm(r_ab, ord = 2, axis = 1)
       norm_bc = np.linalg.norm(r_bc, ord = 2, axis = 1)
       norm_ac = np.linalg.norm(r_ac, ord = 2, axis = 1)

       x = (1/(2*norm_ab*norm_bc))*(norm_ab**2 + norm_bc**2 - norm_ac**2)
       theta = np.arccos(x)

       Z_a = (1 / (norm_ab**3 * norm_bc))[:,np.newaxis] * (np.sum(r_ab*-r_bc, axis = 1)[:, np.newaxis]*r_ab ) + ( 1 / (norm_ab * norm_bc))[:, np.newaxis] * r_bc
       Z_c = (1 / (norm_ab * norm_bc**3))[:,np.newaxis] * (np.sum(r_ab * -r_bc, axis = 1)[:, np.newaxis]*-r_bc) - ( 1 / (norm_ab * norm_bc))[:, np.newaxis] * r_ab

       factor_potential = angle_constants[:, 1]*(theta - np.radians(angle_constants[:, 0]))
       factors = (1/np.sqrt(1 - x**2))*factor_potential 

       energyAngular = np.sum((factor_potential**2)/(2*angle_constants[:, 1]))

       F_a = factors[:, np.newaxis]*Z_a
       F_c = factors[:, np.newaxis]*Z_c
       #F_b = -F_a - F_c

       force = np.zeros((len(coords), 3))
       np.add.at(force, elts1, F_a)
       np.add.at(force, elts2, -(F_a + F_c))
       np.add.at(force, elts3, F_c)

       return energyAngular, force

def calc_dihedral_force(coord: np.array, dihedral_indices: np.array, dihedral_constants: np.array):
       """Computes dihedral energy and force for all dihedrals in dihedral info
       
       INPUT:
       dihedral_indices: list with e.g. [1 2 3 4] for a dihedral between atoms 1 - 2 - 3 - 4
       dihedral_constants: [C_1 C_2 C_3 C_4] for each dihedral

       OUTPUT:
       dihedralEnergy : total dihedral energy from dihedral potential
       force : total dihedral force on all atoms
       """

       coo_a = coord[dihedral_indices[:, 0] - dihedral_indices[0, 1]]
       coo_b = coord[dihedral_indices[:, 1] - dihedral_indices[0, 1]]
       coo_c = coord[dihedral_indices[:, 2] - dihedral_indices[0, 1]]
       coo_d = coord[dihedral_indices[:, 3] - dihedral_indices[0, 1]]

       C_1 = dihedral_constants[:, 0]
       C_2 = dihedral_constants[:, 1]
       C_3 = dihedral_constants[:, 2]
       C_4 = dihedral_constants[:, 3]

       b1_vecs = coo_b - coo_a
       b2_vecs = coo_c - coo_b
       b3_vecs = coo_d - coo_c
       norms1 = np.linalg.norm(b1_vecs, axis=1)
       norms2 = np.linalg.norm(b2_vecs, axis=1)
       norms3 = np.linalg.norm(b3_vecs, axis=1)

       crosses1 = np.cross(b1_vecs, b2_vecs)
       norms_crosses1 = np.linalg.norm(crosses1, axis=1)
       m = crosses1*(1/norms_crosses1[:,np.newaxis])

       crosses2 = np.cross(b2_vecs, b3_vecs)
       norms_crosses2 = np.linalg.norm(crosses2, axis=1)
       n = crosses2*(1/norms_crosses2[:,np.newaxis])
       cross_cross = np.cross(crosses1, crosses2)
       
       angles = np.arctan2(np.sum(b2_vecs*cross_cross, axis=1) , norms2*np.sum(crosses1*crosses2, axis = 1))
       derivatives = (1/2)*(C_1*-np.sin(angles) + 2*C_2*np.sin(2*angles) - 3*C_3*np.sin(3*angles) + 4*C_4*np.sin(4*angles))
       sin_theta1 = (norms1*norms2)/(norms_crosses1*norms1)
       sin_theta2 = (norms2*norms3)/(norms_crosses2*norms3)

       # Calculating the energy:
       dihedralEnergy = np.sum((1/2)*(C_1*(1 + np.cos(angles)) + C_2*(1- np.cos(2*angles)) + C_3*( 1 + np.cos(3*angles)) + C_4*(1 - np.cos(4*angles))))
 
       force_a =m*(-derivatives*sin_theta1)[:, np.newaxis]
       force_d = n*(derivatives*sin_theta2)[:, np.newaxis]

       vectors_oc = (1/2)*b2_vecs
       norms_oc = (1/4)*norms2**2
       vectors_tc  = -( np.cross(vectors_oc + (1/2)*b3_vecs, force_d) - (1/2)*np.cross(b1_vecs, force_a) )
       force_c = np.cross(vectors_tc , vectors_oc)*1/norms_oc[:, np.newaxis]
       force_b = -(force_a + force_c + force_d)

       force = np.zeros(shape=(len(coord), 3))
       np.add.at(force, dihedral_indices[:, 0] - dihedral_indices[0, 1], force_a )
       np.add.at(force, dihedral_indices[:, 1] - dihedral_indices[0, 1], force_b )
       np.add.at(force, dihedral_indices[:, 2] - dihedral_indices[0, 1], force_c )
       np.add.at(force, dihedral_indices[:, 3] - dihedral_indices[0, 1], force_d )

       return dihedralEnergy, force

def lennardJonesForce(nr_atoms: int, coord: np.array, L, r, sigma, epsilon, blockMatrix_neighbours):
    """Computes the Lennard Jones energy and force on all atoms. 

    INPUT:
    nr_atoms : nr of atoms in entire system
    coord : xyz coordinates of each atom
    L : box size for periodic boundary conditions
    r : cutoff radius

    OUTPUT:
    energyLJ : total energy from LJ potential
    force : total LJ force on each atom
    """

    # vectors between atoms i -> j, using closest version of j with periodic boundary condition
    rng = np.arange(1, nr_atoms)
    idx1 = np.repeat(rng - 1, rng[::-1])
    idx2 = np.arange(nr_atoms*(nr_atoms - 1)//2) + np.repeat(nr_atoms - np.cumsum(rng[::-1]), rng[::-1])
    rij_ = np.take(coord, idx2, axis=0) - np.take(coord, idx1, axis=0)

    rij_atoms = np.empty([nr_atoms, nr_atoms, 3])
    rij_atoms[idx1, idx2] = np.mod(rij_ + L/2, L) - L/2 

    #calculate distances between mapped atoms (a x a)
    dist_atoms = np.linalg.norm(rij_atoms, axis = 2)

    #only consider atoms with r_ij < r and not in same molecule
    dist_atoms = sparse.triu(dist_atoms < r).multiply(1 - blockMatrix_neighbours).multiply(dist_atoms)

    #compute LJ force on atoms of neighbours
    I, J, V = sparse.find(dist_atoms)
    div_dist = sparse.csr_array((1/V, (I, J)), shape = [nr_atoms, nr_atoms])
    del I
    del J
    del V

    pw12 = div_dist.multiply(sigma).power(12)
    fact = pw12 - pw12.sqrt()
    magnitudeFactor = 24*epsilon.multiply((div_dist.power(2))).multiply(- fact - pw12) #very slow!!!!!!

    energyLJ = 2*4*epsilon.multiply(fact) #upper triangular so *2

    energyLJ = energyLJ.sum(axis = None)

    del sigma
    del epsilon
    del div_dist
    del fact

    rij_atoms = magnitudeFactor.reshape((-1, 1)).multiply(np.reshape(rij_atoms, (-1, 3)))
    del magnitudeFactor

    #reshape to [nr_atoms, nr_atoms, 3]
    rij_atoms = np.array(rij_atoms.todense())
    rij_atoms = rij_atoms.reshape([nr_atoms, nr_atoms, 3])
    
    #sum over all force on each atom
    force = np.sum(rij_atoms - np.transpose(rij_atoms, (1, 0, 2)), axis = 1)

    return energyLJ, force

def calc_total_force(nr_atoms: int, nr_water: int, coords: np.array, bond_indices: np.array, bond_constants: np.array, angle_indices: np.array, angle_constants: np.array, dihedral_indices: np.array, dihedral_constants: np.array, L, r, sigma, epsilon, blockMatrix_neighbours):
    """Computes total force on each atom and corresponding energy, with the functions in this file
    For this project, this consists of the LJ, bond, angular and dihedral force

    OUTPUT:
    atom_force : total force on each atom
    """

    LJenergy, atom_force = lennardJonesForce(nr_atoms, coords, L, r, sigma, epsilon, blockMatrix_neighbours) 
    bondEnergy = calc_bond_force(coords, bond_indices, bond_constants)

    atom_force += bondEnergy[1]
    bondEnergy = bondEnergy[0]

    angularEnergy = calc_angular_force(coords, angle_indices, angle_constants)
    atom_force += angularEnergy[1]
    angularEnergy = angularEnergy[0]

    dihedralEnergy = calc_dihedral_force(coords[nr_water*3:], dihedral_indices, dihedral_constants)
    atom_force[nr_water*3:] += dihedralEnergy[1]

    return atom_force, LJenergy, bondEnergy, angularEnergy, dihedralEnergy

def thermometer(velocities: np.array, atom_masses: np.array, nr_atoms: int):
       """Measures the kinetic energy and temperature in the system
       
       INPUT:
       velocities : xyz velocity of each atom
       atom_masses : mass of each atom
       nr_atoms : total nr of atoms in system
       
       OUTPUT :
       obtained_temperature : temperature of system
       kinetic_energy : total kinetic energy of system
       """

       constantFact = 1.38065*6.02214*10**(-3)
       kinetic_energy_times2 = np.sum( (np.linalg.norm(velocities, axis = 1)**2)*atom_masses)

       #degrees of freedom equal to 3*N since we only base kinetic energy on velocity in xyz direction
       obtained_temperature = (kinetic_energy_times2)/(3*nr_atoms*constantFact)
       return obtained_temperature, (1/2)*kinetic_energy_times2


def thermostat(velocities: np.array, atom_masses: np.array, nr_atoms: int, temp: float):
       """Computes rescaling factor to have constant temperature
       
       INPUT:
       velocities : xyz velocity of each atom
       atom_masses : mass of each atom
       nr_atoms : total nr of atoms in system
       temp : desired temperature of system

       OUTPUT;
       rescaling_constant : rescaling factor for all velocities
       """
       
       measured_temp = thermometer(velocities, atom_masses, nr_atoms)[0]
       rescaling_constant =  np.sqrt(temp / measured_temp)
       return rescaling_constant

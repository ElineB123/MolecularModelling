import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import time

def initialize_radial(nr_water, from_molecules, atoms_info):
    """Compute the row and column indices, belonging to the atoms which we need to consider in the radial distribution function.
    
    INPUT:  
    from_molecules : [ 0 1 ] to denote from which molecule we want to determine the distribution. 0 water, 1 ethanol
    atoms_info :  [ H O H H O H  ...] list containing the atom type of each atom.
    
    OUTPUT:
    Four arrays of indices, containing the indices of:
        - rows_O where the O atoms are, of molecules of type given in first elt of from_molecules
        - cols_O, where the O atoms are, of molecules of type given in second elt of from_molecules
        - cols_H, where the H atoms are, of molecules of type given in second elt of from_molecules
    """
  
    rows_O = np.squeeze( np.argwhere( atoms_info == 'O' ))    
    cols_O = np.squeeze( np.argwhere( atoms_info == 'O' ))
    cols_H = np.squeeze( np.argwhere( atoms_info == 'H' ))

    if from_molecules[0]:
        rows_O = rows_O [rows_O >= 3*nr_water ]
    else:
        rows_O =  rows_O [rows_O < 3*nr_water ]

    if from_molecules[1]:
        cols_O =  cols_O[ cols_O >= 3*nr_water ]
        cols_H =  cols_H[ cols_H >= 3*nr_water ]
    else:
        cols_O = cols_O[ cols_O < 3*nr_water]
        cols_H = cols_H[ cols_H < 3*nr_water]

    return rows_O, cols_O, cols_H

def radial_r_ij_distribution_EE(rows_measure_O, cols_measure_O, cols_measure_H, r_ij_distances):
    """Computes the number of particles within each "sub radius".
    
    Note, the radial r_ijribution function can be computed easily from this, by dividing the number of particles by (3*nr_water + 9*nr_ethanol)*4*pi*density """

    
    max_r = 10
    delta_r = (1/10)
    values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)

    rad_r_ij_O = np.zeros( shape = len(values_r), dtype = np.float32 )
    rad_r_ij_H = np.zeros( shape = len(values_r), dtype = np.float32 )
    
    for i in range(len(values_r)):
        for row_ind in rows_measure_O:
            colsH_current = np.setdiff1d(cols_measure_H, [row_ind + 1, row_ind - 1, row_ind - 2, row_ind - 4, row_ind -5, row_ind -6])
            rad_r_ij_O[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , cols_measure_O]) & (r_ij_distances[row_ind , cols_measure_O] < values_r[i])))
            rad_r_ij_H[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , colsH_current]) & (r_ij_distances[row_ind , colsH_current] < values_r[i])))    

    return rad_r_ij_O/len(rows_measure_O), rad_r_ij_H/len(rows_measure_O)

def radial_r_ij_distribution_WW(rows_measure_O, cols_measure_O, cols_measure_H, r_ij_distances):
    """Computes the number of particles within each "sub radius".
    
    Note, the radial r_ijribution function can be computed easily from this, by dividing the number of particles by (3*nr_water + 9*nr_ethanol)*4*pi*density """

    
    max_r = 10
    delta_r = (1/10)
    values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)

    rad_r_ij_O = np.zeros( shape = len(values_r), dtype = np.float32 )
    rad_r_ij_H = np.zeros( shape = len(values_r), dtype = np.float32 )
    
    for i in range(len(values_r)):
        for row_ind in rows_measure_O:
            colsH_current = np.setdiff1d(cols_measure_H, [row_ind + 1, row_ind + 2])
            rad_r_ij_O[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , cols_measure_O]) & (r_ij_distances[row_ind , cols_measure_O] < values_r[i])))
            rad_r_ij_H[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , colsH_current]) & (r_ij_distances[row_ind , colsH_current] < values_r[i])))    

    return rad_r_ij_O/len(rows_measure_O), rad_r_ij_H/len(rows_measure_O)


# Compute rdf for water-water
max_r = 10
delta_r = (1/10)
values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)

directory_WW = "./waterSimulation.xyz"
nr_ethanol = 0
nr_water = 900
nr_atoms = int(nr_water*3)
info_atoms = np.asarray(["O", "H", "H"]*nr_water)

rowsO, colsO, colsH = initialize_radial(nr_water, [0, 0], info_atoms)

nr_runs = 500
delta_runs = 100
begin_runs = 0

rdfOResults = np.zeros(shape = (nr_runs, len(values_r)),  dtype = np.float32)
rdfHResults = np.zeros(shape = (nr_runs, len(values_r)),  dtype = np.float32)

for i in range(nr_runs):
    print('i = ', i)
    t0 = time.time()
    df1= pd.read_csv(directory_WW, skiprows=2 + i*100*(nr_atoms + 2) + begin_runs*(nr_atoms + 2), nrows = nr_atoms, header=None, delimiter='\t')
    array = df1.to_numpy()
    coord =  np.array(array[: , 1:], dtype=np.float64)
    normsTraj = sc.spatial.distance.cdist(coord, coord, metric='euclidean')    
    rdfOResults[i], rdfHResults[i] = radial_r_ij_distribution_WW(rowsO, colsO, colsH, normsTraj)
    print('this took ', time.time() - t0)


np.savetxt("rdf_WW_O.csv", rdfOResults, delimiter=',')
np.savetxt("rdf_WW_H.csv", rdfHResults, delimiter=',')

print('Water pure done!')


# Compute rdf for ethanol-ethanol
max_r = 10
delta_r = (1/10)
values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)

directory_EE = "./finalEthanolSimulation.xyz"
nr_ethanol = 277
nr_water = 0
nr_atoms = int(nr_ethanol*9)
ethanol_atoms = np.asarray(["C", "H", "H", "H", "C", "H", "H", "O", "H"]*nr_ethanol)

rowsO, colsO, colsH = initialize_radial(nr_water, [1, 1], ethanol_atoms)

nr_runs = 500
delta_runs = 100
begin_runs = 0

rdfOResults = np.zeros(shape = (nr_runs, len(values_r)),  dtype = np.float32)
rdfHResults = np.zeros(shape = (nr_runs, len(values_r)),  dtype = np.float32)

for i in range(nr_runs):
    print('i = ', i)
    t0 = time.time()
    df1= pd.read_csv(directory_EE, skiprows=2 + i*100*(nr_atoms + 2) + begin_runs*(nr_atoms + 2), nrows = nr_atoms, header=None, delimiter='\t')
    array = df1.to_numpy()
    coord =  np.array(array[: , 1:], dtype=np.float64)
    normsTraj = sc.spatial.distance.cdist(coord, coord, metric='euclidean')    
    rdfOResults[i], rdfHResults[i] = radial_r_ij_distribution_EE(rowsO, colsO, colsH, normsTraj)
    print('this took ', time.time() - t0)


np.savetxt("rdf_etheth_O.csv", rdfOResults, delimiter=',')
np.savetxt("rdf_etheth_H.csv", rdfHResults, delimiter=',')

print('Ethanol pure done!')


# Compute rdf mixture, only for ethanol-water

def radial_r_ij_distribution_mix(rows_measure_O, cols_measure_O, cols_measure_H, r_ij_distances):
    """Computes the number of particles within each "sub radius".
    
    Note, the radial r_ijribution function can be computed easily from this, by dividing the number of particles by (3*nr_water + 9*nr_ethanol)*4*pi*density """

    
    max_r = 10
    delta_r = (1/10)
    values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)

    rad_r_ij_O = np.zeros( shape = len(values_r), dtype = np.float32 )
    rad_r_ij_H = np.zeros( shape = len(values_r), dtype = np.float32 )
    
    for i in range(len(values_r)):
        for row_ind in rows_measure_O:
            colsH_current = cols_measure_H
            rad_r_ij_O[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , cols_measure_O]) & (r_ij_distances[row_ind , cols_measure_O] < values_r[i])))
            rad_r_ij_H[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , colsH_current]) & (r_ij_distances[row_ind , colsH_current] < values_r[i])))    

    return rad_r_ij_O/len(rows_measure_O), rad_r_ij_H/len(rows_measure_O)

max_r = 10
delta_r = (1/10)
values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)

directory_E_mix = "./mixture/EthanolSimulateSystem.xyz"
directory_W_mix = "./mixture/WaterSimulateSystem.xyz"
nr_water = int(864)
nr_ethanol = int(97)
nr_atoms = int(nr_ethanol*9 + nr_water*3)
ethanol_atoms = np.asarray(["C", "H", "H", "H", "C", "H", "H", "O", "H"]*nr_ethanol)
water_atoms = np.asarray(["O", "H", "H"]*nr_water)
atom_types = np.concatenate((water_atoms , ethanol_atoms))

rowsO, colsO, colsH = initialize_radial(nr_water, [1, 0], atom_types)

nr_runs = 500
delta_runs = 100
begin_runs = 0

rdfOResults_mix = np.zeros(shape = (nr_runs, len(values_r)),  dtype = np.float32)
rdfHResults_mix = np.zeros(shape = (nr_runs, len(values_r)),  dtype = np.float32)

for i in range(nr_runs):
    print('i = ', i)
    t0 = time.time()
    df1= pd.read_csv(directory_W_mix, skiprows=2 + i*100*(nr_water*3 + 2) + begin_runs*(nr_water*3 + 2), nrows = nr_water*3, header=None, delimiter='\t')
    df2= pd.read_csv(directory_E_mix, skiprows=2 + i*100*(nr_ethanol*9 + 2) + begin_runs*(nr_ethanol*9 + 2), nrows = nr_ethanol*9, header=None, delimiter='\t')
    array1 = df1.to_numpy()
    array2 = df2.to_numpy()
    coord1 =  np.array(array1[: , 1:], dtype=np.float64)
    coord2 =  np.array(array2[: , 1:], dtype=np.float64)
    coord = np.concatenate((coord1, coord2))
    normsTraj = sc.spatial.distance.cdist(coord, coord, metric='euclidean')    
    rdfOResults_mix[i], rdfHResults_mix[i] = radial_r_ij_distribution_mix(rowsO, colsO, colsH, normsTraj)
    print('this took ', time.time() - t0)


np.savetxt("rdf_mix_O.csv", rdfOResults_mix, delimiter=',')
np.savetxt("rdf_mix_H.csv", rdfHResults_mix, delimiter=',')

print('Mixture done!')

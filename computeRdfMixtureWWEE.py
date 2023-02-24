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



def radial_r_ij_distribution_mix_water(rows_measure_O_WW, cols_measure_O_WW, cols_measure_H_WW, r_ij_distances):
    """Computes the number of particles within each "sub radius".
    
    Note, the radial r_ijribution function can be computed easily from this, by dividing the number of particles by (3*nr_water + 9*nr_ethanol)*4*pi*density """

    
    max_r = 10
    delta_r = (1/10)
    values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)

    rad_r_ij_O_WW = np.zeros( shape = len(values_r), dtype = np.float32 )
    rad_r_ij_H_WW = np.zeros( shape = len(values_r), dtype = np.float32 )

    
    for i in range(len(values_r)):
        # Water - Water
        for row_ind in rows_measure_O_WW:
            colsH_current = np.setdiff1d(cols_measure_H_WW, [row_ind +1, row_ind + 2])
            rad_r_ij_O_WW[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , cols_measure_O_WW]) & (r_ij_distances[row_ind , cols_measure_O_WW] < values_r[i])))
            rad_r_ij_H_WW[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , colsH_current]) & (r_ij_distances[row_ind , colsH_current] < values_r[i])))    

    return rad_r_ij_O_WW/len(rows_measure_O_WW), rad_r_ij_H_WW/len(rows_measure_O_WW)


def radial_r_ij_distribution_mix_ethanol(rows_measure_O_EE, cols_measure_O_EE, cols_measure_H_EE, r_ij_distances):
    """Computes the number of particles within each "sub radius".
    
    Note, the radial r_ijribution function can be computed easily from this, by dividing the number of particles by (3*nr_water + 9*nr_ethanol)*4*pi*density """

    
    max_r = 10
    delta_r = (1/10)
    values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)


    rad_r_ij_O_EE = np.zeros( shape = len(values_r), dtype = np.float32 )
    rad_r_ij_H_EE = np.zeros( shape = len(values_r), dtype = np.float32 )
    
    for i in range(len(values_r)):
        # Ethanol - Ethanol
        for row_ind in rows_measure_O_EE:
            colsH_current = np.setdiff1d(cols_measure_H_EE, [row_ind +1, row_ind - 1, row_ind - 2, row_ind - 4, row_ind - 5, row_ind - 6])
            rad_r_ij_O_EE[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , cols_measure_O_EE]) & (r_ij_distances[row_ind , cols_measure_O_EE] < values_r[i])))
            rad_r_ij_H_EE[i] += np.sum( ((values_r[i] - delta_r < r_ij_distances[row_ind , colsH_current]) & (r_ij_distances[row_ind , colsH_current] < values_r[i])))    

    return rad_r_ij_O_EE/len(rows_measure_O_EE), rad_r_ij_H_EE/len(rows_measure_O_EE)

max_r = 10
delta_r = (1/10)
values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)

nr_water = int(864)
nr_ethanol = int(97)
nr_atoms = int(nr_ethanol*9 + nr_water*3)
ethanol_atoms = np.asarray(["C", "H", "H", "H", "C", "H", "H", "O", "H"]*nr_ethanol)
water_atoms = np.asarray(["O", "H", "H"]*nr_water)
atom_types = np.concatenate((water_atoms , ethanol_atoms))

runs_begin = 400
nr_runs = 500
delta_runs = 100
begin_runs = 0

max_r = 10
delta_r = (1/10)
values_r = np.arange(start= delta_r, stop = max_r + delta_r, step = delta_r)


# rdf of water-water of the mixture: 
directory_W_mix = "./mixture/WaterSimulateSystem.xyz"
rowsO_EW, colsO_EW, colsH_EW = initialize_radial(nr_water, [1, 0], atom_types)
rowsO_WW, colsO_WW, colsH_WW = initialize_radial(nr_water, [0, 0], atom_types)
rowsO_EE, colsO_EE, colsH_EE = initialize_radial(nr_water, [1, 1], atom_types)
rdf_WW_mix_O = np.zeros(shape = (nr_runs - runs_begin , len(values_r)),  dtype = np.float32)
rdf_WW_mix_H = np.zeros(shape = (nr_runs - runs_begin, len(values_r)),  dtype = np.float32)

dfWater= pd.read_csv(directory_W_mix, skiprows=2 + runs_begin*100*(nr_water*3 + 2), header=None, delimiter='\t')
dfWater = dfWater.dropna(axis = 0, how = 'any')
dfWater = dfWater.to_numpy()
dfWater = np.array(dfWater[: , 1:], dtype=np.float64)

for i in range(100):
    waterCoords = dfWater[i*100*3*nr_water : 100*i*3*nr_water + 3*nr_water]
    print(np.shape(waterCoords))

    normsTraj = sc.spatial.distance.cdist(waterCoords, waterCoords, metric='euclidean')    
    rdf_WW_mix_O[i], rdf_WW_mix_H[i] = radial_r_ij_distribution_mix_water(rowsO_WW, colsO_WW, colsH_WW, normsTraj)


np.savetxt("rdf_mix_O_WW.csv", rdf_WW_mix_O, delimiter=',')
np.savetxt("rdf_mix_H_WW.csv", rdf_WW_mix_H, delimiter=',')


# Computer rdf Ethanol-Ethanol of the mixture :
# Due to memory issues, do it in two parts
directory_E_mix = "./mixture/EthanolSimulateSystem.xyz"
rowsO_EE, colsO_EE, colsH_EE = initialize_radial(nr_water, [1, 1], atom_types) 
rowsO_EE, colsO_EE, colsH_EE = rowsO_EE -nr_water*3 - 1, colsO_EE-nr_water*3 - 1, colsH_EE-nr_water*3 - 1
rdf_EE_mix_O = np.zeros(shape = (nr_runs - runs_begin , len(values_r)),  dtype = np.float32)
rdf_EE_mix_H = np.zeros(shape = (nr_runs - runs_begin , len(values_r)),  dtype = np.float32)


# Part I
dfEth= pd.read_csv(directory_E_mix, skiprows=2 + runs_begin*100*(nr_ethanol*9 + 2), header=None, nrows = 50*100*(nr_ethanol*9 + 2), delimiter='\t')
dfEth = dfEth.dropna(axis = 0, how = 'any')
dfEth = dfEth.to_numpy()
dfEth = np.array(dfEth[: , 1:], dtype=np.float64)
print('loaded')

for i in range(50):
    ethCoords = dfEth[i*100*9*nr_ethanol : 100*i*9*nr_ethanol + 9*nr_ethanol]
    print('i = ', i)
    normsTraj = sc.spatial.distance.cdist(ethCoords, ethCoords, metric='euclidean')    
    rdf_EE_mix_O[i], rdf_EE_mix_H[i] = radial_r_ij_distribution_mix_ethanol(rowsO_EE, colsO_EE, colsH_EE, normsTraj)


# Part II:
dfEth= pd.read_csv(directory_E_mix, skiprows=2 + runs_begin*100*(nr_ethanol*9 + 2) + 50*100*(nr_ethanol*9 + 2), header=None, delimiter='\t')
dfEth = dfEth.dropna(axis = 0, how = 'any')
dfEth = dfEth.to_numpy()
dfEth = np.array(dfEth[: , 1:], dtype=np.float64)
print('loaded')

for i in range(50):
    ethCoords = dfEth[i*100*9*nr_ethanol : 100*i*9*nr_ethanol + 9*nr_ethanol]
    print('i = ', i + 50)
    normsTraj = sc.spatial.distance.cdist(ethCoords, ethCoords, metric='euclidean')    
    rdf_EE_mix_O[i + 50], rdf_EE_mix_H[i + 50] = radial_r_ij_distribution_mix_ethanol(rowsO_EE, colsO_EE, colsH_EE, normsTraj)

np.savetxt("rdf_mix_O_EE.csv", rdf_EE_mix_O, delimiter=',')
np.savetxt("rdf_mix_H_EE.csv", rdf_EE_mix_H, delimiter=',')





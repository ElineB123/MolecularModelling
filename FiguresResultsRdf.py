import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

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

################################################################################################################################################
# Water - water
################################################################################################################################################
nr_water = 900
nr_atoms = int(nr_water*3)
water_atoms = np.asarray(['O', 'H', 'H']*nr_water)
colsO_WW, colsH_WW = initialize_radial(nr_water, [0, 0], water_atoms)[1:]

dens_O_WW = len(colsO_WW)/(30**3)
dens_H_WW = (len(colsH_WW ) - 1)/(30**3)

max_r = 10
delta_r = (1/10)
values_r = np.arange(start=delta_r, stop = max_r + delta_r, step = delta_r)

ResO_WW = np.loadtxt("./rdf_waterwater_O.csv", delimiter=',')
ResH_WW = np.loadtxt("./rdf_waterwater_H.csv", delimiter=',')

frameBegin = 400

resultsO = np.asarray(ResO_WW)[frameBegin:][1: ]/((values_r**2)*4*np.pi*dens_O_WW*delta_r) 
resultsH = np.asarray(ResH_WW)[frameBegin:][1: ]/((values_r**2)*4*np.pi*dens_H_WW*delta_r) 


lowerCIO = np.zeros(len(values_r))
lowerCIH = np.zeros(len(values_r))
meansO_WW = np.zeros(len(values_r))
meansH_WW = np.zeros(len(values_r))
upperCIO = np.zeros(len(values_r))
upperCIH = np.zeros(len(values_r))
for frame_nr in range(100):
    avgO = np.mean( resultsO[:, frame_nr])
    meansO_WW[frame_nr] = avgO
    sdO = np.std(resultsO[:, frame_nr])
    lowerCIO[frame_nr] = avgO - 1.96*sdO/99
    upperCIO[frame_nr] = avgO + 1.96*sdO/99

    avgH = np.mean( resultsH[:, frame_nr])
    meansH_WW[frame_nr] = avgH
    sdH = np.std(resultsH[:, frame_nr])
    lowerCIH[frame_nr] = avgH - 1.96*sdH/99
    upperCIH[frame_nr] = avgH + 1.96*sdH/99

# fig, ax = plt.subplots(figsize=(9,6))
# plt.plot(values_r , meansO, label = 'OO')
# ax.fill_between(values_r, lowerCIO, upperCIO, color='b', alpha=.5)
# plt.plot(values_r , meansH, label = 'OH')
# ax.fill_between(values_r, lowerCIH, upperCIH, color='b', alpha=.5)
# ax.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('r (Å)', fontsize = 18)
# plt.ylabel('g(r)', fontsize = 18)
# plt.legend(fontsize = 18)
# plt.show()

################################################################################################################################################
# Ethanol-Ethanol
################################################################################################################################################

nr_water = 0
nr_ethanol = 277
nr_atoms = int(nr_ethanol*9)
ethanol_atoms = np.asarray(["C", "H", "H", "H", "C", "H", "H", "O", "H"]*nr_ethanol)
colsO, colsH = initialize_radial(nr_water, [1, 1], ethanol_atoms)[1:]

dens_O_EE = len(colsO)/(30**3)
dens_H_EE = (len(colsH ) - 5)/(30**3)

max_r = 10
delta_r = (1/10)
values_r = np.arange(start=delta_r, stop = max_r + delta_r, step = delta_r)

ResO_EE = np.loadtxt("./rdf_etheth_O.csv", delimiter=',')
ResH_EE = np.loadtxt("./rdf_etheth_H.csv", delimiter=',')

frameBegin = 400

resultsO = np.asarray(ResO_EE)[frameBegin:][1: ]/((values_r**2)*4*np.pi*dens_O_EE*delta_r) 
resultsH = np.asarray(ResH_EE)[frameBegin:][1: ]/((values_r**2)*4*np.pi*dens_H_EE*delta_r) 

lowerCIO = np.zeros(len(values_r))
lowerCIH = np.zeros(len(values_r))
meansO = np.zeros(len(values_r))
meansH = np.zeros(len(values_r))
upperCIO = np.zeros(len(values_r))
upperCIH = np.zeros(len(values_r))
for frame_nr in range(100):
    avgO = np.mean( resultsO[:, frame_nr])
    meansO[frame_nr] = avgO
    sdO = np.std(resultsO[:, frame_nr])
    lowerCIO[frame_nr] = avgO - 1.96*sdO/99
    upperCIO[frame_nr] = avgO + 1.96*sdO/99

    avgH = np.mean( resultsH[:, frame_nr])
    meansH[frame_nr] = avgH
    sdH = np.std(resultsH[:, frame_nr])
    lowerCIH[frame_nr] = avgH - 1.96*sdH/99
    upperCIH[frame_nr] = avgH + 1.96*sdH/99

# fig, ax = plt.subplots(figsize=(9,6))
# plt.plot(values_r , meansO, label = 'OO')
# ax.fill_between(values_r, lowerCIO, upperCIO, color='b', alpha=.5)
# plt.plot(values_r , meansH, label = 'OH')
# ax.fill_between(values_r, lowerCIH, upperCIH, color='b', alpha=.5)
# ax.tick_params(axis='both', which='major', labelsize=18)
# plt.xlabel('r (Å)', fontsize = 18)
# plt.ylabel('g(r)', fontsize = 18)
# plt.legend(fontsize = 18)
# plt.show()


################################################################################################################################################
# Mixture
################################################################################################################################################


nr_water = int(864)
nr_ethanol = int(97)
nr_atoms = int(nr_ethanol*9 + nr_water*3)
ethanol_atoms = np.asarray(["C", "H", "H", "H", "C", "H", "H", "O", "H"]*nr_ethanol)
water_atoms = np.asarray(["O", "H", "H"]*nr_water)
atom_types = np.concatenate((water_atoms , ethanol_atoms))

rowsO, colsO, colsH = initialize_radial(nr_water, [1, 0], atom_types)

dens_O_Mix = len(colsO)/(30**3)
dens_H_Mix = (len(colsH))/(30**3)

max_r = 10
delta_r = (1/10)
values_r = np.arange(start=delta_r, stop = max_r + delta_r, step = delta_r)

ResO = np.loadtxt("./rdf_mix_O.csv", delimiter=',')
ResH = np.loadtxt("./rdf_mix_H.csv", delimiter=',')

frameBegin = 400

resultsO = np.asarray(ResO)[frameBegin:][1: ]/((values_r**2)*4*np.pi*dens_O_Mix*delta_r) 
resultsH = np.asarray(ResH)[frameBegin:][1: ]/((values_r**2)*4*np.pi*dens_H_Mix*delta_r) 

lowerCIO = np.zeros(len(values_r))
lowerCIH = np.zeros(len(values_r))
meansO_EW = np.zeros(len(values_r))
meansH_EW = np.zeros(len(values_r))
upperCIO = np.zeros(len(values_r))
upperCIH = np.zeros(len(values_r))
for frame_nr in range(100):
    avgO = np.mean( resultsO[:, frame_nr])
    meansO_EW[frame_nr] = avgO
    sdO = np.std(resultsO[:, frame_nr])
    lowerCIO[frame_nr] = avgO - 1.96*sdO/99
    upperCIO[frame_nr] = avgO + 1.96*sdO/99

    avgH = np.mean( resultsH[:, frame_nr])
    meansH_EW[frame_nr] = avgH
    sdH = np.std(resultsH[:, frame_nr])
    lowerCIH[frame_nr] = avgH - 1.96*sdH/99
    upperCIH[frame_nr] = avgH + 1.96*sdH/99


# Now water-water:
rowsO, colsO, colsH = initialize_radial(nr_water, [0, 0], atom_types)
dens_O_Mix = len(colsO)/(30**3)
dens_H_Mix = (len(colsH))/(30**3)

max_r = 10
delta_r = (1/10)
values_r = np.arange(start=delta_r, stop = max_r + delta_r, step = delta_r)

ResO = np.loadtxt("./rdf_mix_O_WW.csv", delimiter=',')
ResH = np.loadtxt("./rdf_mix_H_WW.csv", delimiter=',')


resultsO = np.asarray(ResO)[1: ]/((values_r**2)*4*np.pi*dens_O_Mix*delta_r) 
resultsH = np.asarray(ResH)[1: ]/((values_r**2)*4*np.pi*dens_H_Mix*delta_r) 

print(resultsH)

lowerCIO = np.zeros(len(values_r))
lowerCIH = np.zeros(len(values_r))
meansO_EW = np.zeros(len(values_r))
meansH_EW = np.zeros(len(values_r))
upperCIO = np.zeros(len(values_r))
upperCIH = np.zeros(len(values_r))
for frame_nr in range(100):
    avgO = np.mean( resultsO[:, frame_nr])
    meansO_EW[frame_nr] = avgO
    sdO = np.std(resultsO[:, frame_nr])
    lowerCIO[frame_nr] = avgO - 1.96*sdO/99
    upperCIO[frame_nr] = avgO + 1.96*sdO/99

    avgH = np.mean( resultsH[:, frame_nr])
    meansH_EW[frame_nr] = avgH
    sdH = np.std(resultsH[:, frame_nr])
    lowerCIH[frame_nr] = avgH - 1.96*sdH/99
    upperCIH[frame_nr] = avgH + 1.96*sdH/99




# Ethanol-Ethanol
rowsO, colsO, colsH = initialize_radial(nr_water, [1, 1], atom_types) 

ResO = np.loadtxt("./rdf_mix_O_EE.csv", delimiter=',')
ResH = np.loadtxt("./rdf_mix_O_EE.csv", delimiter=',')

dens_O_Mix = len(colsO)/(30**3)
dens_H_Mix = (len(colsH - 6))/(30**3)

resultsO = np.asarray(ResO)[1: ]/((values_r**2)*4*np.pi*dens_O_Mix*delta_r) 
resultsH = np.asarray(ResH)[1: ]/((values_r**2)*4*np.pi*dens_H_Mix*delta_r) 


lowerCIO = np.zeros(len(values_r))
lowerCIH = np.zeros(len(values_r))
meansO_EE = np.zeros(len(values_r))
meansH_EE = np.zeros(len(values_r))
upperCIO = np.zeros(len(values_r))
upperCIH = np.zeros(len(values_r))
for frame_nr in range(100):
    avgO = np.mean( resultsO[:, frame_nr])
    meansO_EE[frame_nr] = avgO
    sdO = np.std(resultsO[:, frame_nr])
    lowerCIO[frame_nr] = avgO - 1.96*sdO/99
    upperCIO[frame_nr] = avgO + 1.96*sdO/99

    avgH = np.mean( resultsH[:, frame_nr])
    meansH_EE[frame_nr] = avgH
    sdH = np.std(resultsH[:, frame_nr])
    lowerCIH[frame_nr] = avgH - 1.96*sdH/99
    upperCIH[frame_nr] = avgH + 1.96*sdH/99



fig, ax = plt.subplots(figsize=(9,6))
plt.plot(values_r , meansO_WW, label = 'OO Water-Water')
plt.plot(values_r , meansO_EW, label = 'OO Ethanol-Water')
plt.plot(values_r , meansO_EE, label = 'OO Ethanol-Ethanol')
plt.plot(values_r , meansH_WW, label = 'OH Water-Water')
plt.plot(values_r , meansH_EW, label = 'OH Ethanol-Water')
plt.plot(values_r , meansH_EE, label = 'OH Ethanol-Ethanol')
ax.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel('r (Å)', fontsize = 18)
plt.ylabel('g(r)', fontsize = 18)
plt.legend(fontsize = 18)
plt.show()

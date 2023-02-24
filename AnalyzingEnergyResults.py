import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

directoryEnergyWW = './results_energy/energyResultsWW.csv'
directoryEnergyEE = './results_energy/energyResultsEE.csv'
directoryEnergyMix = './results_energy/energyResultsMix.csv'
directory_EE_dihedral = './dihedralEnergyEth.csv'

directoryWW_TempKinAfter = './results_energy/TemperatureKineticAfterWW.txt'
directoryWW_TempKinBefore = './results_energy/TemperatureKineticBeforeWW.txt'
directoryEE_TempKinAfter = './results_energy/TemperatureKineticAfterEE.txt'
directoryEE_TempKinBefore = './results_energy/TemperatureKineticBeforeEE.txt'
directoryMix_TempKinAfter = './results_energy/TemperatureKineticAfterMix.txt'
directoryMix_TempKinBefore = './results_energy/TemperatureKineticBeforeMix.txt'


dfWW= pd.read_csv(directoryEnergyWW, header=None, delimiter=',')
dfWW = dfWW.to_numpy()
energyBondWW, energyAngleWW, energyDihedralWW, energyLJWW = dfWW[:, 0], dfWW[:, 1], dfWW[:, 2], dfWW[:, 3]
potentialWW = energyBondWW + energyAngleWW + energyDihedralWW + energyLJWW

dfEE= pd.read_csv(directoryEnergyEE, header=None, delimiter=',')
dfEE = dfEE.to_numpy()
energyBondEE, energyAngleEE, energyDihedralEE, energyLJEE = dfEE[:, 0], dfEE[:, 1], dfEE[:, 2], dfEE[:, 3]
energyDihedralEE = np.squeeze(pd.read_csv(directory_EE_dihedral, header=None, delimiter=',').to_numpy())
potentialEE = energyBondEE + energyAngleEE + energyDihedralEE + energyLJEE


dfmix= pd.read_csv(directoryEnergyMix, header=None, delimiter=',')
dfmix = dfmix.to_numpy()
energyBondMix, energyAngleMix, energyDihedralMix, energyLJMix = dfmix[:, 0], dfmix[:, 1], dfmix[:, 2], dfmix[:, 3]
potentialMix = energyBondMix + energyAngleMix + energyDihedralMix + energyLJMix


dfWWtempKin_before = pd.read_csv(directoryWW_TempKinBefore, skiprows = 1, header=None, delimiter='\t')
dfWWtempKin_after = pd.read_csv(directoryWW_TempKinAfter, skiprows = 1, header=None, delimiter='\t')
dfEEtempKin_before = pd.read_csv(directoryEE_TempKinBefore, skiprows = 1, header=None, delimiter='\t')
dfEEtempKin_after = pd.read_csv(directoryEE_TempKinAfter, skiprows = 1, header=None, delimiter='\t')
dfMixtempKin_before = pd.read_csv(directoryMix_TempKinBefore, skiprows = 1, header=None, delimiter='\t')
dfMixtempKin_after = pd.read_csv(directoryMix_TempKinAfter, skiprows = 1, header=None, delimiter='\t')

dfWWtempKin_before = np.asarray(dfWWtempKin_before.dropna(how = 'any', axis=0).to_numpy(), dtype = np.float64)
dfWWtempKin_after = np.asarray(dfWWtempKin_after.dropna(how = 'any', axis=0).to_numpy(), dtype = np.float64)
dfEEtempKin_before = np.asarray(dfEEtempKin_before.dropna(how = 'any', axis=0).to_numpy(), dtype = np.float64)
dfEEtempKin_after = np.asarray(dfEEtempKin_after.dropna(how = 'any', axis=0).to_numpy(), dtype = np.float64)
dfMixtempKin_before = np.asarray(dfMixtempKin_before.dropna(how = 'any', axis=0).to_numpy(), dtype = np.float64)
dfMixtempKin_after = np.asarray(dfMixtempKin_after.dropna(how = 'any', axis=0).to_numpy(), dtype = np.float64)

times = np.arange(start = 0, step = 0.02, stop = 50000*0.02)
warmup = 500


# Potential split out for the systems
fig, ax = plt.subplots(figsize=(9,6))
plt.plot(times[warmup:], energyAngleWW[warmup:], label = 'Angular Energy')
plt.plot(times[warmup:], energyBondWW[warmup:], label = 'Bond Energy')
plt.plot(times[warmup:], energyDihedralWW[warmup:], label = 'Dihedral Energy')
# plt.plot(times[warmup:], energyLJWW[warmup:], label = 'Lennard Jones')
plt.xlabel('Time (ps)', fontsize = 18)
plt.ylabel('Energy (kJ)', fontsize = 18)
plt.legend(loc = 4, fontsize = 18)
# plt.title('Pure Water')
ax.tick_params(axis='both', which='major', labelsize=18)
plt.show()


fig, ax = plt.subplots(figsize=(9,6))
plt.plot(times[warmup:], energyAngleEE[warmup:], label = 'Angular Energy')
plt.plot(times[warmup:], energyBondEE[warmup:], label = 'Bond Energy')
plt.plot(times[warmup:], energyDihedralEE[warmup:], label = 'Dihedral Energy')
# plt.plot(times[warmup:], energyLJEE[warmup:], label = 'Lennard Jones')
plt.xlabel('Time (ps)', fontsize = 18)
plt.ylabel('Energy (kJ)', fontsize = 18)
plt.legend(fontsize = 18)
# plt.title('Pure Ethanol', fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(9,6))
plt.plot(times[warmup:], energyAngleMix[warmup:], label = 'Angular Energy')
plt.plot(times[warmup:], energyBondMix[warmup:], label = 'Bond Energy')
plt.plot(times[warmup:], energyDihedralMix[warmup:], label = 'Dihedral Energy')
# plt.plot(times[warmup:], energyLJMix[warmup:], label = 'Lennard Jones')
plt.xlabel('Time (ps)', fontsize = 18)
plt.ylabel('Energy (kJ)', fontsize = 18)
plt.legend(loc = 4, fontsize = 18)
# plt.title('Mixture Ethanol - Water')
ax.tick_params(axis='both', which='major', labelsize=18)
plt.show()

# Lennard Jones energy :
fig, ax = plt.subplots(figsize=(9,6))
plt.plot(times[warmup:], energyLJWW[warmup:], label = 'Pure Water')
plt.plot(times[warmup:], energyLJEE[warmup:], label = 'Pure Ethanol')
plt.plot(times[warmup:], energyLJMix[warmup:], label = 'Mixture Ethanol-Water')
plt.xlabel('Time (ps)', fontsize = 18)
plt.ylabel('Energy (kJ)', fontsize = 18)
plt.legend(fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.show()

# Bond energy:
times = np.arange(start = 0, step = 0.02, stop = 50000*0.02)
print(times)
plt.plot(times[warmup:], energyBondWW[warmup:], label = 'Pure Water')
plt.plot(times[warmup:], energyBondEE[warmup:], label = 'Pure Ethanol')
plt.plot(times[warmup:], energyBondMix[warmup:], label = 'Mixture Ethanol-Water')
plt.xlabel('time (ps)')
plt.legend()
plt.title('Bond Energy')
# plt.show()

# Angular energy:
times = np.arange(start = 0, step = 0.02, stop = 50000*0.02)
print(times)
plt.plot(times[warmup:], energyAngleWW[warmup:], label = 'Pure Water')
plt.plot(times[warmup:], energyAngleEE[warmup:], label = 'Pure Ethanol')
plt.plot(times[warmup:], energyAngleMix[warmup:], label = 'Mixture Ethanol-Water')
plt.xlabel('time (ps)')
plt.legend()
plt.title('Angular Energy')
# plt.show()

# Dihedral energy:
times = np.arange(start = 0, step = 0.02, stop = 50000*0.02)
print(times)
plt.plot(times[warmup:], energyDihedralWW[warmup:], label = 'Pure Water')
plt.plot(times[warmup:], energyDihedralEE[warmup:], label = 'Pure Ethanol')
plt.plot(times[warmup:], energyDihedralMix[warmup:], label = 'Mixture Ethanol-Water')
plt.xlabel('time (ps)')
plt.legend()
plt.title('Dihedral Energy')
# plt.show()

# Potential energy : 
plt.plot(times[warmup:], potentialWW[warmup:], label = 'Pure Water')
plt.plot(times[warmup:], potentialEE[warmup:], label = 'Pure Ethanol')
plt.plot(times[warmup:], potentialMix[warmup:], label = 'Mixture Ethanol-Water')
plt.xlabel('time (ps)')
plt.legend()
plt.title('Potential Energy')
plt.show()

### Plots Temperature : 
fig, ax = plt.subplots(figsize=(9,6))
plt.plot(times[warmup:], dfWWtempKin_before[: , 0][warmup:])
plt.plot(times[warmup:], dfWWtempKin_after[: , 0][warmup:])
plt.xlabel('Time (ps)', fontsize = 18)
plt.ylabel('Temperature (K)', fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(9,6))
plt.plot(times[warmup:], dfEEtempKin_before[: , 0][warmup:])
plt.plot(times[warmup:], dfEEtempKin_after[: , 0][warmup:])
plt.xlabel('Time (ps)', fontsize = 18)
plt.ylabel('Temperature (K)', fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(9,6))
plt.plot(times[warmup:], dfMixtempKin_before[: , 0][warmup:])
plt.plot(times[warmup:], dfMixtempKin_after[: , 0][warmup:])
plt.xlabel('Time (ps)', fontsize = 18)
plt.ylabel('Temperature (K)', fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.show()



# ### Plots Kinetic before and after:
fig, ax = plt.subplots(figsize=(9,6))
plt.plot(times[warmup:], dfWWtempKin_before[: , 1][warmup:], label = 'Water before')
plt.plot(times[warmup:], dfWWtempKin_after[: , 1][warmup:], label = 'Water after')
plt.plot(times[warmup:], dfEEtempKin_before[: , 1][warmup:], label = 'Ethanol before')
plt.plot(times[warmup:], dfEEtempKin_after[: , 1][warmup:], label = 'Ethanol after')
plt.plot(times[warmup:], dfMixtempKin_before[: , 1][warmup:], label = 'Mixture before')
plt.plot(times[warmup:], dfMixtempKin_after[: , 1][warmup:], label = 'Mixture after', c = 'black')
plt.xlabel('Time (ps)', fontsize= 18)
plt.ylabel('Energy (kJ)', fontsize = 18)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.legend(fontsize = 18)
plt.show()


### Plots main groups of energy - kin, pot and total
fig, ax = plt.subplots()
plt.plot(times[warmup:], dfWWtempKin_before[: , 1][warmup:], label = 'Kinetic energy')
# plt.plot(times[warmup:], dfWWtempKin_after[: , 1][warmup:], label = 'Fixed kinetic energy')
plt.plot(times[warmup:], potentialWW[warmup:], label = 'Potential energy')
plt.plot(times[warmup:], potentialWW[warmup:] + dfWWtempKin_before[:, 1][warmup: ], label = 'Total energy')
plt.title('Pure Water', fontsize = 16)
plt.legend(fontsize = 16)
plt.xlabel('Time (ps)', fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()


fig, ax = plt.subplots()
plt.plot(times[warmup:], dfEEtempKin_before[: , 1][warmup:], label = 'Kinetic energy')
# plt.plot(times[warmup:], dfWWtempKin_after[: , 1][warmup:], label = 'Fixed kinetic energy')
plt.plot(times[warmup:], potentialEE[warmup:], label = 'Potential energy')
plt.plot(times[warmup:], potentialEE[warmup:] + dfEEtempKin_before[:, 1][warmup: ], label = 'Total energy')
plt.title('Pure Ethanol', fontsize = 16)
plt.legend(fontsize = 16)
plt.xlabel('Time (ps)', fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()


fig, ax = plt.subplots()
plt.plot(times[warmup:], dfMixtempKin_before[: , 1][warmup:], label = 'Kinetic energy')
# plt.plot(times[warmup:], dfWWtempKin_after[: , 1][warmup:], label = 'Fixed kinetic energy')
plt.plot(times[warmup:], potentialMix[warmup:], label = 'Potential energy')
plt.plot(times[warmup:], potentialMix[warmup:] + dfMixtempKin_before[:, 1][warmup: ], label = 'Total energy')
plt.title('Mixture Ethanol - Water', fontsize = 16)
plt.legend(loc = 1, fontsize = 16)
plt.xlabel('Time (ps)', fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()







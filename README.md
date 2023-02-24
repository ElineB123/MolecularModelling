To simulate the MD system, code has been written from scratch using only the numpy and scipy
libraries. Int his file, we will describe the structure of the code. For a flowchart of the code, see Figure 6. 
For an overall idea of the code, note that we decided against object oriented programming, and instead performed 
all our computations in vectorized format. All results and variables, for example the XYZ positions of all atoms, 
or the forces acting on each atom, are therefore defined in matrix shape.

The code for the simulation makes use of three different files; initialization.py, computations.py
and running.py. To construct the partial radial distribution functions, as well as the figures on
the temperature and energies, a separate file results.py was created. This file uses the information
saved from the entire simulation, to create these results.

In initialization.py, there are six functions that are in various degrees required to setting up the
system. 
First, there are two functions readXYZfile and readTopology, which read the XYZ
positions and the topology respectively from input files. These input files are supplied in this repository.
In the topology file, the integers in the first columns denote the index number of the corresponding atoms.
Important is to note that the units used throughout this simulation are
• Distance:  ̊A,
• Energy: kJ,
• Time: 0.1 ps,
• Angles: rad,
• Mass: amu.

Aside from this, the functions initializeVelocity and generateMolecules in initialization.py initialize
the velocities and positions of the required number of water and ethanol molecules. The
former randomly generates initial velocity vectors for all atoms taken from a normal distribution.
The latter function places the molecules randomly on a grid within the desired box, with enough
distance between each. The relative positions of the different molecules are obtained from input
XYZ files. Furthermore, it also generates the corresponding information on the bonds, angles
and dihdredals of all created atoms. 
Lastly, the functions initializeSigmaEps and cmMatrix create matrices used to compute the LJ force
and the center of mass respectively. These matrices are constant, and can therefore be initialized
before the simulation is started.

Once the starting positions and the corresponding matrices have been initialized, simulateSystem
in running.py can be started. This function runs the MD simulation with the help of the func-
tions defined in the computations.py file. For example, to determine the total force acting
on each atom, functions computing the bond, angular, dihedral and LJ force are combined in
calcTotalForce in computations.py. Additionally, the temperature and thermostat are also
measured in the computations.py file. When the required computations have been performed,
the positions and velocities are updated with Velocity Verlet in running.py. There is no separate
function defined, it is implemented within simulateSystem. Once the coordinates have been
updated, we remap the coordinates back inside the box, based on the center of mass of the
molecule. After this remapping, the velocities are updated and the simulation is ready for a new
round. Every round of the simulation, the coordinates of all atoms, as well as the energies and temperature are written in files.

The results from the files were analyzed with the AnalyzingEnergyResults.py and ... files.

A flowchart of the code can be found in this repository as well.

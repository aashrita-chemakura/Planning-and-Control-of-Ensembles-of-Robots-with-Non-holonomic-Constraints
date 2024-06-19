
# Swarm Robotics Simulation Study

## Overview
This repository contains the Python code for simulating the planning and control of ensembles of robots with non-holonomic constraints. The project aims to demonstrate the dynamics of robot swarms with an emphasis on path planning and behavior under randomized initial conditions. The simulation visualizes the outcomes using graphs and animated GIFs.
## Team Members

| Name                    | GitHub Profile                             |
|-------------------------|--------------------------------------------|
| Saketh Narayan Banagiri | [Sakethbngr](https://github.com/Sakethbngr)|
| Aashrita Chemakura      | [aashrita-chemakura](https://github.com/aashrita-chemakura) |


## Setup and Execution
Follow these instructions to set up and run the simulation:
### Prerequisites
Ensure you have Python installed on your machine along with the following libraries:

- numpy: For handling large, multi-dimensional arrays and matrices.
- math: Provides access to mathematical functions.
- matplotlib: Useful for creating static, interactive, and animated visualizations in Python.
- imageio: For reading and writing a wide range of image data, including animated images, video, and volumetric data.
- cvxopt: A package for convex optimization ([download here](https://cvxopt.org/)).
## Installation
1. Clone this repository to your local machine:
```bash
git clone https://github.com/aashrita-chemakura/Planning-and-Control-of-Ensembles-of-Robots-with-Non-holonomic-Constraints.git
```
2. Install the required Python libraries:
```bash
pip install numpy matplotlib imageio
pip install cvxopt
```
### Running the Simulation
1. Navigate to the project directory:
 ```bash
cd Planning-and-Control-of-Ensembles-of-Robots-with-Non-holonomic-Constraints
```
2. Execute the main.py script:
```bash
python main.py
```
### Outputs
- The script will generate several plots and a GIF animation illustrating the swarm behavior.
- All outputs will be saved in the 'results' folder.

## Challenges Encountered
The main challenge encountered was the inability to achieve the desired final state consistently across all swarm configurations. This was primarily due to the robots being initialized at random positions. Although the algorithm was designed to handle a variety of scenarios, it was not robust enough to guarantee the desired outcome in every instance. Despite these challenges, the results obtained provide a valuable approximation of the expected behaviors and are instrumental for further research and enhancements.

## References
For further understanding and in-depth technical details, consult the following:
- [CVXOPT Library Documentation](https://cvxopt.org/documentation/index.html)

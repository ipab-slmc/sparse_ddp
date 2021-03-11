# Sparsity-Inducing Optimal Control via Differential Dynamic Programming

This repository contains scripts and notebooks to reproduce the results presented in:

> Traiko Dinev∗, Wolfgang Merkt∗, Vladimir Ivan, Ioannis Havoutis, Sethu Vijayakumar. **Sparsity-Inducing Optimal Control via Differential Dynamic Programming**. Proc. IEEE International Conference on Robotics and Automation (ICRA 2021), Xian, China, 2021.

## Installation

This repository depends on the following ROS packages whose dependencies can be installed from binaries:

  - [exotica](https://github.com/ipab-slmc/exotica)
  - [exotica_satellite_dynamics_solver](https://github.com/ipab-slmc/exotica_satellite_dynamics_solver)

<!-- We also provide a Dockerfile to make installation easier. -->

## Examples

  - `roslaunch exotica_satellite_dynamics_solver example.launch` - will run trajectory optimisation for the example in Fig. 6.

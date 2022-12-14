# How PINNs cheat: Predicting chaotic motion of a double pendulum

## Abstract
Despite extensive research, physics-informed neural networks (PINNs) are still difficult to train, especially when the optimization relies heavily on the physics loss term. Convergence problems frequently occur when simulating dynamical systems with high-frequency components, chaotic or turbulent behavior. 
In this work, we discuss whether the traditional PINN framework is able to predict chaotic motion by conducting experiments on the undamped double pendulum. 
Our results demonstrate that PINNs do not exhibit any sensitivity to perturbations in the initial condition. Instead, the PINN optimization consistently converges to physically correct solutions that violate the initial condition only marginally, but diverge significantly from the desired solution due to the chaotic nature of the system. In fact, the PINN predictions primarily exhibit low-frequency components with a smaller magnitude of higher-order derivatives, which favors lower physics loss values compared to the desired solution. 
We thus hypothesize that the PINNs "cheat" by shifting the initial conditions to values that correspond to physically correct solutions that are easier to learn.
Initial experiments suggest that domain decomposition combined with an appropriate loss weighting scheme mitigates this effect and allows convergence to the desired solution.

## Instructions

The repository contains all code needed to reproduce the results of this work. 


#### Requirements

```
tensorflow 2.3.0
numpy 1.21.2
scipy 1.7.1
matplotlib 3.4.3
yaml 6.0
json 2.0.9
notebook 6.4.5
```


#### Usage

* Run and train PINNs to simulate the double pendulum: `main.ipynb`
* Evaluate the results and analyze the PINN prediction compared to RK45: `analysis.ipynb`


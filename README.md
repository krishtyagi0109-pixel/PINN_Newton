# PINN for Newton's Second Law

A beginner **Physics-Informed Neural Network (PINN)** that solves Newton's second law as an ordinary differential equation, built with [DeepXDE](https://github.com/lululxvi/deepxde) and PyTorch.

> Built as a learning project while getting started with PINNs.

## Problem

The network learns the displacement `u(t)` of a mass `m` under a force `F(t)`:

```
m · u''(t) = F(t)
```

with initial conditions:

```
u(0)  = 1    (initial displacement)
u'(0) = 0    (initial velocity)
```

In this version `m = 1` and `F(t) = 0`. You can edit the `force(t)` function to try other forces.

## How it works

The network is trained without any data. The loss combines:

1. **ODE residual**: `m·u'' − F(t)` evaluated at 200 collocation points in time
2. **Initial conditions**: errors in `u(0)` and `u'(0)`

Velocity and acceleration are computed with automatic differentiation using `dde.grad.jacobian` and `dde.grad.hessian`.

| Setting      | Value                    |
|--------------|--------------------------|
| Network      | FNN `[1, 32, 32, 32, 1]` |
| Activation   | tanh                     |
| Initializer  | Glorot normal            |
| Optimizer    | Adam, lr = 0.001         |
| Iterations   | 10,000                   |
| Time domain  | t ∈ [0, 10]              |

## Installation

```bash
git clone https://github.com/krishtyagi0109-pixel/PINN_Newton.git
cd PINN_Newton
pip install -r requirements.txt
```

## Results

### Displacement


![Displacement](results/displacement.png)



### Velocity


![Velocity](results/velocity.png)



### Acceleration


![Acceleration](results/acceleration.png)



## Project structure

```
PINN_Newton/
├── pinn_newton.py
├── requirements.txt
├── results/
│   ├── displacement.png
│   ├── velocity.png
│   └── acceleration.png
└── README.md
```

## Ideas for extension

- [ ] Constant and time-dependent forces, e.g. `F(t) = sin(t)`
- [ ] Spring force (harmonic oscillator)
- [ ] Damping and driven oscillators
- [ ] Compare against analytical solutions and report the error
- [ ] Inverse problems (infer an unknown force from data)

## Acknowledgements

- [DeepXDE](https://github.com/lululxvi/deepxde) by Lu Lu et al.
- Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks* (2019)

## License

This project is licensed under the MIT License.

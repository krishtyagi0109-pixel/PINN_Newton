import os

import torch

import numpy as np

import pandas as pd

import torch.nn as nn

os.environ["DDE_BACKEND"] = "pytorch"

!pip install deepxde

import deepxde as dde

m = 1.0

def force(t):
  return 0.0


def equation(t, u):

    velocity = dde.grad.jacobian(u, t)

    acceleration = dde.grad.hessian(u, t)

    return m * acceleration - force(t)

geom = dde.geometry.TimeDomain(0,10)

def initial_displacement(t):
  return 1

def initial_velocity(t):
  return 0

ic_u = dde.icbc.IC(
    geom,
    initial_displacement,
    lambda _,on_initial: on_initial
    
)

ic_v = dde.icbc.OperatorBC(
    geom,
    lambda t,u, _: dde.grad.jacobian(u,t),
    lambda _, on_initial:on_initial 

)

data = dde.data.PDE(
    geom,
    eqn,
    [ic_u, ic_v],
    num_domain=200,
    num_boundary=2,
    num_test=100
)

net = dde.nn.FNN(
    [1,32,32,32,1],
    "tanh",
    "Glorot normal"
)

model = dde.Model(data,net)

model.compile("adam", lr=0.001)

losshistory, train_state = model.train(
    iterations=10000
)




t_test = np.linspace(0, 10, 200)[:, None]

u_pred = model.predict(t_test)

print(u_pred)

v_pred = model.predict(
    t_test,
    operator = lambda t,u: dde.grad.jacobian(u,t)
)

print(v_pred)


a_pred = model.predict(
    t_test,
    operator=lambda t, u: dde.grad.hessian(u, t)
)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t_test, u_pred)
plt.ylabel("Displacement")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_test, v_pred)
plt.ylabel("Velocity")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_test, a_pred)
plt.ylabel("Acceleration")
plt.xlabel("Time")
plt.grid()

plt.tight_layout()
plt.show()


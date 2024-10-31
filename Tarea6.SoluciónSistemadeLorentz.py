# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:12:21 2024

@author: Karla Edith
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

def solve_lorenz(state, t, σ, ρ, β):
    x, y, z = state
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = (x * y) - (β * z)
    return np.array([dx, dy, dz])

initial_state = (1., 1., 1.)
σ, ρ, β = 10, 28, 8/3
t = np.linspace(0, 100, 3000)
states = odeint(solve_lorenz, initial_state, t, args=(σ, ρ, β))

xs, ys, zs = states[:, 0], states[:, 1], states[:, 2]
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xs, ys, zs, lw=0.5, c='#7B86B2')
ax.set_title("Lorenz Attractor with odeint")
plt.show()

cmap = plt.cm.plasma

def lorenz_attractor(state0, parameters, ax=None, text_offset=0.01):
    σ, ρ, β = parameters
    n = 3000
    t = np.linspace(0, 100, n)
    states = odeint(solve_lorenz, state0, t, args=(σ, ρ, β))
    xs, ys, zs = states[:, 0], states[:, 1], states[:, 2]

    if ax is None:
        ax = plt.figure(figsize=(7, 8)).add_subplot(projection='3d')

    s = 10
    for i in range(0, n-s, s):
        ax.plot(xs[i:i+s+1], ys[i:i+s+1], zs[i:i+s+1], color=cmap(i/n), alpha=0.5, lw=1)

    ax.text2D(0.44, text_offset, f"ρ={ρ}", transform=ax.transAxes)
    ax.figure.tight_layout()
    return ax.figure

rho_values = [-42, 15, 28, 100]
solutions = [lorenz_attractor((1, 1, 1), (10, rho, 8/3)) for rho in rho_values]

for solution in solutions:
    plt.show()

rows, cols = 2, 2
size = (10, 10)
fig, axs = plt.subplots(rows, cols, figsize=size, subplot_kw=dict(projection='3d'))
rho_values2d = np.array(rho_values).reshape(rows, cols)

for row, col in np.ndindex((rows, cols)):
    rho = rho_values2d[row, col]
    lorenz_attractor((1, 1, 1), (10, rho, 8/3), ax=axs[row, col], text_offset=-0.03)

plt.show()

fsize = (15, 15)
ndim = 8

def butterfly(ax, colormap, rho, angle):
    tmax = 100
    n = 10000
    sigma, rho, beta = (10, rho, 2.667)
    u0, v0, w0 = (0, 1, 1.05)
    t = np.linspace(0, tmax, n)
    soln = odeint(solve_lorenz, (u0, v0, w0), t, args=(sigma, rho, beta))
    x, y, z = soln[:, 0], soln[:, 1], soln[:, 2]

    ax.set_facecolor('k')
    s = 10
    cmap = getattr(plt.cm, colormap)
    for i in range(0, n-s, s):
        ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/n), alpha=0.4)

    ax.set_axis_off()
    ax.view_init(angle, angle)

axs = plt.figure(facecolor='k', figsize=fsize).subplots(ndim, ndim, subplot_kw={'projection': '3d'})
plt.show()

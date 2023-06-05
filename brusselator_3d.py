#!/usr/bin/env python3

import ufl

import numpy as np

from dolfinx import io, mesh
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem import Function, FunctionSpace

from mpi4py import MPI
from petsc4py import PETSc
from ufl import dot, dx, grad

# Define temporal parameters
t = 0  # Start time
T = 30  # Final time
num_steps = 300
dt = T / num_steps  # time step size

A = 1
B = 3

Dc = 1  # diffusion coefficient for c
Dmu = 0.1  # diffusion coefficient for mu

msh = mesh.create_box(
    comm=MPI.COMM_WORLD,
    points=[(0, 0, 0), (64, 64, 64)],
    n=[16, 16, 16],
    cell_type=mesh.CellType.tetrahedron,
)

P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
ME = FunctionSpace(msh, P1 * P1)

print("FE space generated")

q, v = ufl.TestFunctions(ME)

u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

c, mu = ufl.split(u)  # split vector into components u = (c, mu)
c0, mu0 = ufl.split(u0)

# zero-out array
u.x.array[:] = 0.0

# i.c.
u.sub(0).interpolate(lambda x: A + 0.0 * x[0])
u.sub(1).interpolate(
    lambda x: B / A + 0.1 * A * np.random.standard_normal(size=x.shape[1])
)


u.x.scatter_forward()

react_c = A - (1 + B) * c + mu * c * c
react_mu = B * c - mu * c * c


F0 = (
    c * q * dx
    + Dc * dt * dot(grad(c), grad(q)) * dx
    - c0 * q * dx
    - dt * react_c * q * dx
)
F1 = (
    mu * v * dx
    + Dmu * dt * dot(grad(mu), grad(v)) * dx
    - mu0 * v * dx
    - dt * react_mu * v * dx
)
F = F0 + F1

problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-10

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

file = io.XDMFFile(MPI.COMM_WORLD, "/mnt/d/experimental_results/bruss3d_output.xdmf", "w")

file.write_mesh(msh)
t = 0.0

c = u.sub(0)
mu = u.sub(1)
u0.x.array[:] = u.x.array

file.write_function(c, t)
file.write_function(mu, t)

while t < T:
    t += dt
    r = solver.solve(u)
    print(f"Step {int(t/dt)}: num iterations: {r[0]}")
    if not r[1]:
        print("Newton iteration failed to converge! Exiting")
        break
    u0.x.array[:] = u.x.array
    file.write_function(c, t)
    file.write_function(mu, t)

file.close()

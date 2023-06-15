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

# http://hopf.chem.brandeis.edu/members_content/yanglingfa/pattern/oreg/index.htmls
# Define temporal parameters
t = 0  # Start time
T = 500  # Final time
num_steps = 8000
dt = (T - t) / num_steps  # time step size


Dy = 3  # diffusion coefficient for y
Dz = 100  # diffusion coefficient for z
Dr = 0.1  # diffusion coefficient for r

qpar = 0.01
eps = 0.5
f = 0.8

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[(0, 0), (128, 128)],
    n=[64, 64],
    cell_type=mesh.CellType.quadrilateral,
)
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
ME = FunctionSpace(msh, ufl.MixedElement(P1, P1, P1))

v, q, w = ufl.TestFunctions(ME)

u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

y, z, r = ufl.split(u)  # split vector into components u = (y, z, r)
y0, z0, r0 = ufl.split(u0)


# zero-out array
u.x.array[:] = 0.0

# i.c.
u.sub(0).interpolate(lambda xi: 1 + 0.0 * xi[0])
u.sub(1).interpolate(lambda xi: 1 + 0.1 * np.random.standard_normal(size=xi.shape[1]))
u.sub(2).interpolate(lambda xi: 1 + 0.1 * np.random.standard_normal(size=xi.shape[1]))
u.x.scatter_forward()

react_y = (1 / eps) * (y - y * y - f * z * ((y - qpar) / (y + qpar)) - 0.5 * (y - r))
react_z = y - z
react_r = (1 / (2 * eps)) * (y - r)

F0 = (
    y * v * dx
    - y0 * v * dx
    + Dy * dt * dot(grad(y), grad(v)) * dx
    - dt * react_y * v * dx
)
F1 = (
    z * q * dx
    - z0 * q * dx
    + Dz * dt * dot(grad(z), grad(q)) * dx
    - dt * react_z * q * dx
)
F2 = (
    r * w * dx
    - r0 * w * dx
    + Dr * dt * dot(grad(r), grad(w)) * dx
    - dt * react_r * w * dx
)
F = F0 + F1 + F2

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

file = io.XDMFFile(MPI.COMM_WORLD, "oreg_output.xdmf", "w")

file.write_mesh(msh)

y = u.sub(0)
z = u.sub(1)
r = u.sub(2)
u0.x.array[:] = u.x.array

file.write_function(y, t)
file.write_function(z, t)
file.write_function(r, t)

while t < T:
    t += dt
    w = solver.solve(u)
    print(f"Step {int(t/dt)}: num iterations: {w[0]}")
    if not w[1]:
        print("Newton iteration failed to converge! Exiting")
        break
    u0.x.array[:] = u.x.array
    file.write_function(y, t)
    file.write_function(z, t)
    file.write_function(r, t)

file.close()

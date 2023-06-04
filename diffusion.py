import ufl

import numpy as np

from mpi4py import MPI

from dolfinx import io
from dolfinx import fem
from dolfinx import mesh
from dolfinx.fem import FunctionSpace

from petsc4py import PETSc

# Define temporal parameters
t = 0  # Start time
T = 1.5  # Final time
num_steps = 3500
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-2, -2]), np.array([2, 2])],
    [nx, ny],
    mesh.CellType.triangle,
)

V = fem.FunctionSpace(domain, ("Lagrange", 1))


def initial_condition(x, a=10):
    return a * np.exp(-(x[0] ** 2 + x[1] ** 2))


u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)  # all boundary nodes are accepted t


rhs = PETSc.ScalarType(0)

xdmf = io.XDMFFile(domain.comm, "diffusion_output.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)


u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, rhs)
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = fem.petsc.assemble_matrix(bilinear_form)
A.assemble()
b = fem.petsc.create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)


for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
        print(loc_b)
    fem.petsc.assemble_vector(b, linear_form)

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)
xdmf.close()

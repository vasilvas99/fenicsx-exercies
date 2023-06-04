import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh
from dolfinx.fem import Function, FunctionSpace
import ufl
# Create mesh
comm = MPI.COMM_WORLD
mesh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)

# Define function space
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

# Define parameters
Du = 0.1
Dv = 0.1
k = 0.9
f = 2.0
dt = 0.01
T = 10.0

# Define initial conditions
u_0 = dolfinx.fem.Function(P1)
u_0.interpolate(lambda x: 1 + 0.1 * np.sin(x[0] * np.pi) * np.sin(x[1] * np.pi))

v_0 = dolfinx.fem.Function(P1)
v_0.interpolate(lambda x: 3.0)

# Define variational problem
u, v = dolfinx.fem.Function(P1), dolfinx.Function(P1)
u_n, v_n = dolfinx.fem.Function(P1), dolfinx.Function(P1)

f = dolfinx.Expression(f"f + u*u*v - (k+1)*u", f=f, k=k, degree=2)

dx = u.function_space().measure
ds = u.function_space().mesh.xdim - 1

F = u * v_n * dx - u_n * v_n * dx / dt \
    + Du * dolfinx.inner(dolfinx.grad(u), dolfinx.grad(u_n)) * v_n * dx \
    - f * v_n * dx

F += v * u_n * v_n * dx - v_n * u_n * v_n * dx / dt \
    + Dv * dolfinx.inner(dolfinx.grad(v), dolfinx.grad(v_n)) * u_n * dx \
    + f * v_n * dx

# Create time-stepping function
t = 0
while t <= T + dolfinx.DOLFIN_EPS:
    # Set up linear problem
    A = dolfinx.fem.assemble_matrix(F, bcs=[])
    b = dolfinx.fem.create_vector(A)
    dolfinx.fem.assemble_vector(b, F)

    # Apply Dirichlet boundary conditions
    u_bc = dolfinx.DirichletBC(u.function_space(), u_0, lambda x, on_boundary: on_boundary)
    u_bc.apply(A)
    u_bc.apply(b)

    v_bc = dolfinx.DirichletBC(v.function_space(), v_0, lambda x, on_boundary: on_boundary)
    v_bc.apply(A)
    v_bc.apply(b)

    # Solve linear problem
    solver = PETSc.KSP().create(comm)
    solver.setFromOptions()
    solver.setOperators(A)
    solver.solve(b, u.vector)
    solver.solve(b, v.vector)

    # Save solution to file (optional)
    dolfinx.io.XDMFFile(comm, "brusselator_u.xdmf").write(u)
    dolfinx.io.XDMFFile(comm, "brusselator_v.xdmf").write(v)

    # Update previous solution
    u_n.vector.set_local(u.vector.get_local())
    v_n.vector.set_local(v.vector.get_local())

    # Increment time
    t += dt

import ufl

from mpi4py import MPI

from dolfinx import io
from dolfinx import fem
from dolfinx import mesh
from dolfinx.fem import FunctionSpace

from petsc4py.PETSc import ScalarType


domain = mesh.create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, mesh.CellType.tetrahedron)
V = FunctionSpace(domain, ("CG", 1))

uD = fem.Function(V)
uD.interpolate(lambda x: 1 + 0 * x[0] ** 2 + 0 * x[1] ** 2)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, ScalarType(-6))


a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = fem.petsc.LinearProblem(
    a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()


with io.XDMFFile(domain.comm, "poisson_output.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

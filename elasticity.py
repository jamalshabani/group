from firedrake import *
from firedrake.output import VTKFile

length = 1
width = 0.2
mesh = RectangleMesh(40, 20, length, width)

V = VectorFunctionSpace(mesh, "Lagrange", 1)

mybcs = DirichletBC(V, Constant([0, 0]), 1)

rho = Constant(0.01)
g = Constant(1)
f = as_vector([0, -rho*g])
mu = Constant(1)
lambda_ = Constant(0.25)
Id = Identity(mesh.geometric_dimension()) # 2x2 Identity tensor

def epsilon(u):
    return 0.5*(grad(u) + grad(u).T)

def sigma(u):
    return lambda_*div(u)*Id + 2*mu*epsilon(u)

u = Function(V)
v = TestFunction(V)
a = inner(sigma(u), epsilon(v))*dx
L = inner(f, v)*dx

R = a - L

uh = Function(V)
solve(R == 0, u, bcs = mybcs)

VTKFile("elasticity/solution.pvd").write(u)
# Simple Poisson Equations

from firedrake import *
from firedrake.output import VTKFile
mesh = UnitSquareMesh(100, 100)

# We need to decide on the function space in which we'd like to solve the
# problem. Let's use piecewise linear functions continuous between

V = FunctionSpace(mesh, "CG", 1)

# We'll also need the test and trial functions corresponding to this
# function space::

u = Function(V)
v = TestFunction(V)

# We declare a function over our function space and give it the
# value of our right hand side function::

g = Function(V)
x, y = SpatialCoordinate(mesh)
g.interpolate(Constant(20.0))

# We can now define the bilinear and linear forms for the left and right
# hand sides of our equation respectively::
# 1 is Left 2 is Right 3 is Bottom 4 is Top
mybcs = DirichletBC(V, Constant(0.0), [1, 2, 3])

a = (inner(grad(u), grad(v))) * dx
L = inner(g, v) * ds(4)

R = a - L

solve(R == 0, u, bcs = mybcs)

VTKFile("laplace/solution.pvd").write(u)
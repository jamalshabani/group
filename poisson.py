# # Simple Poisson Equation

from firedrake import *
from firedrake.output import VTKFile
mesh = UnitSquareMesh(100, 100) #2D
#mesh = UnitCubeMesh()

# # We need to decide on the function space in which we'd like to solve the
# # problem. Let's use piecewise linear functions continuous between

V = FunctionSpace(mesh, "CG", 1)

# # We'll also need the test and trial functions corresponding to this
# # function space::

u = TrialFunction(V)
v = TestFunction(V)

# # We declare a function over our function space and give it the
# # value of our right hand side function::

f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate(2*pi**2*sin(pi*x)*sin(pi*y))
# f=2π^2sin(πx)sin(πy)
# u=sin(πx)sin(πy)


# # We can now define the bilinear and linear forms for the left and right
# # hand sides of our equation respectively::
# # 1 is Left 2 is Right 3 is Bottom 4 is Top
myBc = DirichletBC(V, Constant(0.0), [1, 2, 3, 4])

a = (inner(grad(u), grad(v))) * dx
L = inner(f, v) * dx

solution = Function(V)

solve(a == L, solution, bcs = myBc)

VTKFile("poisson/solution.pvd").write(u)

u_exact = Function(V)
u_exact.interpolate(sin(x*pi)*sin(y*pi))
print(sqrt(assemble(inner(u - u_exact, u - u_exact) * dx)))
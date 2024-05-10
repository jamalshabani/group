from firedrake import *
from petsc4py import PETSc

length = 1
width = 0.2
mesh = RectangleMesh(40, 20, length, width)

V = FunctionSpace(mesh, "CG", 1)

bcs = DirichletBC(V, Constant(0.0), [1, 2, 3, 4])

u = Function(V)
u_i = Function(V, name = "Solution")
f = Constant(1.0)
x, y = SpatialCoordinate(mesh)
u.interpolate(Constant(1))

func1 = inner(grad(u), grad(u)) * dx
func2 = inner(u, u) * dx
func3 = inner(f, u) * dx

J = func1 + func2 - func3

v = TestFunction(V)
gradu = Function(V)

a = 2 * inner(grad(gradu), grad(v)) * dx + 2 * inner(grad(gradu), grad(v)) * dx
L = inner(f, v) * dx

R_forward = a - L

test_u = File('test/test.pvd')

def FormObjectiveGradient(tao, x, G):

	i = tao.getIterationNumber()
	if (i%5) == 0:
		# Save output files after each 5 iterations
		u_i.interpolate(u)
		test_u.write(u_i, time = i)

	with u.dat.vec as u_vec:
		u_vec.set(0.0)
		u_vec.axpy(1.0, x)
		
	# Solve gradient weak form
	solve(R_forward == 0, gradu, bcs = bcs)
	with gradu.dat.vec as gradu_vec:
		G.set(0.0)
		G.axpy(1.0, gradu_vec)

	f_val = assemble(J)
	return f_val

# Setting TAO solver
tao = PETSc.TAO().create(PETSc.COMM_SELF)
tao.setType('cg')
tao.setObjectiveGradient(FormObjectiveGradient, None)
tao.setFromOptions()

# Initial design guess
with u.dat.vec as u_vec:
	x = u_vec.copy()

# Solve the optimization problem
tao.solve(x)
tao.destroy()

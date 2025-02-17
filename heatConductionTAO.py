# HOW TO RUN THIS CODE:
# python3 heatConductionTAO.py -tao_monitor -tax_max_it 100
# Option: -tao_monotor prints the Function value at each iteration
# Option: -tao_max_it specify maximum number of iterations. Here we set to 100


"""The aim of the current optimization is to find the best optimal shape that minimizes
the heat when the amount of high-conductivity material is limited. In other words
the final structure optimally dissipates the heat taken from the lower conductivity
material"""

# Mathematically, problem we are solving is:
"""     min \int_{\Omega} fT dx + \alpha\int_{\Omega}|\nabla \rho|^2 dx
        subject to -\text{div}(k(\rho)\nabla T) = f in \Omega
									 T = 0 on \Gamma_D
						 \nabla T \cdot n = 0 on \Gamma_N """

# where k(a) = delta + (1 - delta) * a^2 i.e k(0) = delta, k(1) = 1, delta = 10^{-3}
# k(a) is the thermal conductivity of the domain \Omega
# alpha = 10^{-8} is the penalization parameter
# f = 10^{-2} is the volumetric source term
# \rho is the design variable i.e \rho = 0 means "void" or "hole" and \rho = 1 means "solid" or "material"
# \Omega is the design domain. It is a unit square in this problem

# This problem is self-adjoint. Lucky for us, we do not need to solve the adjoint PDE to compute sensitivity

from firedrake import *
from petsc4py import PETSc

# Generate a mesh
n = 250 # Mesh size
mesh = UnitSquareMesh(n, n)

# Define the function space
V = FunctionSpace(mesh, 'CG', 1)

# Create initial design
###### Begin Initial Design #####
rho = Function(V)
rho_i = Function(V, name = "Material density")
rho = Constant(0.5)
rho = interpolate(rho, V)
###### End Initial Design #####

# Define the constant parameters used in the problem
alpha = 1.0e-8
lagrange = 0.00015
delta = 1.0e-3

# The volumetric source term for the PDE
f = Constant(1.0e-2)

# The thermal conductivity of the solid
def k(rho):
    return delta + (1 - delta) * rho ** 2

# Define test function and beam displacement
v = TestFunction(V)
T = Function(V, name = "Temperature")

# The left and top side has T = 0
bcs = DirichletBC(V, Constant(0.0), [1, 4])

# Define the objective function
func1 = inner(f, T) * dx
func2 = alpha * inner(grad(rho), grad(rho)) * dx
func3 = lagrange * rho * dx  # Volume constraint

J = func1 + func2 + func3

# Define the weak form for forward PDE
a_forward = inner(k(rho) * grad(T), grad(v)) * dx
L_forward = inner(f, v) * dx
R_fwd = a_forward - L_forward

# Define the Lagrangian
a_legrange = inner(k(rho) * grad(T), grad(T)) * dx
L_legrange = inner(f, T) * dx
R_legrange = a_legrange - L_legrange
L = J - R_legrange

beam = File('heatConduction/square.pvd')

def FormObjectiveGradient(tao, x, G):

	i = tao.getIterationNumber()
	if (i%5) == 0:
		# Save output files after each 5 iterations
		rho_i.interpolate(rho)
		beam.write(rho_i, T, time = i)

	with rho.dat.vec as rho_vec:
		rho_vec.set(0.0)
		rho_vec.axpy(1.0, x)

	volume = assemble(rho * dx)
	print("The volume fraction is {}".format(volume))
	print(" ")

	# Solve forward PDE
	solve(R_fwd == 0, T, bcs = bcs)

	dJdrho = assemble(derivative(L, rho))
	with dJdrho.dat.vec as dJdrho_vec:
		G.set(0.0)
		G.axpy(1.0, dJdrho_vec)

	f_val = assemble(J)
	return f_val

# Setting lower and upper bounds
lb = Constant(0)
ub = Constant(1)
lb = interpolate(lb, V)
ub = interpolate(ub, V)

with lb.dat.vec as lb_vec:
	rho_lb = lb_vec

with ub.dat.vec as ub_vec:
	rho_ub = ub_vec

# Setting TAO solver
tao = PETSc.TAO().create(PETSc.COMM_SELF)
tao.setType('bncg')
tao.setObjectiveGradient(FormObjectiveGradient, None)
tao.setVariableBounds(rho_lb, rho_ub)
tao.setFromOptions()

# Initial design guess
with rho.dat.vec as rho_vec:
	x = rho_vec.copy()

# Solve the optimization problem
tao.solve(x)
tao.destroy()

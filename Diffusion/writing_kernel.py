from PCGP import PCGP_Builder
import sympy as sp


def B(D, x):
    (dx, ) = D
    return sp.matrices.Matrix([[1],[x[0]**2*dx**2 + dx**2 + 2*x[0]*dx]
                                ])
def base_kernel(x, y):
    A, l = sp.symbols("A l")
    diff = sp.sqrt((x[0]-y[0])**2)
    return A*(1 + sp.sqrt(5)*diff/l + 5*diff**2/(3*l**2))*sp.exp(-diff*sp.sqrt(5)/l)
builder = PCGP_Builder()
builder.add_kernel(B, shared_base_kernel = True)#, base_kernel = base_kernel)
builder.write("Diffusion")


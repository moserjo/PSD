
import sympy
from PCGP import PCGP_Builder

def B(D, x): 
    n = sympy.symbols("n")  
    return sympy.matrices.Matrix([
            [1],
            [x[0]**2*D[0]**2+x[0]*D[0]+(x[0]**2-n**2)],
        ])

builder = PCGP_Builder()
builder.add_kernel(B, shared_base_kernel = True)
builder.write("Bessel")

def B_vanilla(D, x): 
    return sympy.matrices.Matrix([
            [1],
            ])

builder = PCGP_Builder()
builder.add_kernel(B_vanilla, shared_base_kernel = True)
builder.write("Vanilla")
from PCGP import PCGP_Builder
import sympy as sp


def B(D, x):
    return sp.matrices.Matrix([[1],[-D[0]**2 - sp.cos(x[0])]
                                ])

builder = PCGP_Builder()
builder.add_kernel(B, shared_base_kernel = True)
builder.write("Waves")

#def V(x):
          #  return (1/(1.45+torch.tanh(5*(x[0]-0.55))))**2
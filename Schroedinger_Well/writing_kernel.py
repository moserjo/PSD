from PCGP import PCGP_Builder
import sympy as sp


def B(D, x):
    def V(x):
        sp.cos(x)
        #sp.functions.elementary.piecewise.Piecewise((0, (x<0.35 or x > 0.65)), (-120, (x>=0.35 or x <= 0.65) ))
    return sp.matrices.Matrix([[1],[-D[0]**2 + sp.cos(x[0])]
                                ])

def base_kernel(x, y):
    A, l = sp.symbols("A l")
    diff = sp.sqrt((x[0]-y[0])**2)
    return A*(1 + sp.sqrt(5)*diff/l + 5*diff**2/(3*l**2))*sp.exp(-diff*sp.sqrt(5)/l)
   # {\displaystyle C_{5/2}(d)=\sigma ^{2}\left(1+{\frac {{\sqrt {5}}d}{\rho }}+{\frac {5d^{2}}{3\rho ^{2}}}\right)\exp \left(-{\frac {{\sqrt {5}}d}{\rho }}\right).}
builder = PCGP_Builder()
builder.add_kernel(B, shared_base_kernel = True)#, base_kernel = base_kernel)
builder.write("Schroedinger")

"""Schrödinger square-well example from pcgp-figures.

PDE:  -psi''(x) + V(x) psi(x) = 0,  x in [0, 1]
BCs:  psi(0) = 0,  psi(1) = 1

Operator:
  L psi = -psi'' + V(x) psi

Coefficient (potential well):
  V(x) = -120   for x in (0.35, 0.65)   [finite square well of depth 120]
  V(x) =   0    otherwise

Analytical solution (piecewise):
  The ODE -psi'' + V psi = 0 becomes psi'' = V(x) psi.

  Outside the well V = 0  ->  psi'' = 0  ->  linear:
    psi_L(x) = B * x                        for x in [0, 0.35)
    psi_R(x) = E + F * x                    for x in [0.65, 1]

  Inside the well V = -120  ->  psi'' = -120 psi  ->  oscillatory:
    psi_m(x) = C cos(k x) + D sin(k x)     for x in [0.35, 0.65)
    with k = sqrt(120) ≈ 10.9545

  The five constants B, C, D, E, F are fixed by matching psi and psi' at both
  interfaces (x = 0.35 and x = 0.65) and the right BC psi(1) = 1:

  Numerically:
    B ≈ -1.952014
    C ≈  0.412080
    D ≈  0.573334
    E ≈  0.147894
    F ≈  0.852106

Training setup used in pcgp-figures (fig1_zoo):
  Base kernel:        squared-exponential, lengthscale = 0.085
  Collocation nodes:  n_coll = 28, Xc = np.linspace(0, 1, 28)
  Observations:       n_obs  = 7,  drawn at seed = 2
  Observation noise:  sigma  = 0.02
"""

import numpy as np


DEPTH = 1#120.0
WIDTH = 0.3
X_CENTER = 0.5
WELL_A = X_CENTER - WIDTH / 2   # 0.35
WELL_B = X_CENTER + WIDTH / 2   # 0.65
K_IN = np.sqrt(DEPTH)           # sqrt(120) ≈ 10.9545
DOMAIN = (0.0, 1.0)
BC = ((0.0, 0.0), (1.0, 1.0))
SOURCE = 0.0


def _compute_coefficients():
    a, b = WELL_A, WELL_B
    k = K_IN
    # 5x5 linear system for [B, C, D, E, F]:
    #   psi_L(a)  = psi_m(a)  -> B*a  = C*cos(k*a) + D*sin(k*a)
    #   psi_L'(a) = psi_m'(a) -> B    = -k*C*sin(k*a) + k*D*cos(k*a)
    #   psi_m(b)  = psi_R(b)  -> C*cos(k*b) + D*sin(k*b) = E + F*b
    #   psi_m'(b) = psi_R'(b) -> -k*C*sin(k*b) + k*D*cos(k*b) = F
    #   psi_R(1)  = 1          -> E + F = 1
    M = np.zeros((5, 5))
    M[0] = [a,  -np.cos(k*a),          -np.sin(k*a),          0,  0]
    M[1] = [1,   k*np.sin(k*a),        -k*np.cos(k*a),        0,  0]
    M[2] = [0,  -np.cos(k*b),          -np.sin(k*b),          1,  b]
    M[3] = [0,   k*np.sin(k*b),        -k*np.cos(k*b),        0,  1]
    M[4] = [0,   0,                     0,                     1,  1]
    rhs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    return np.linalg.solve(M, rhs)


_B, _C, _D, _E, _F = _compute_coefficients()


def potential(x):
    """Square-well potential."""
    x = np.asarray(x, float)
    return np.where((x > WELL_A) & (x < WELL_B), -DEPTH, 0.0)


def u_exact(x):
    """Analytical piecewise solution."""
    x = np.asarray(x, float)
    psi_L = _B * x
    psi_m = _C * np.cos(K_IN * x) + _D * np.sin(K_IN * x)
    psi_R = _E + _F * x
    return np.where(x < WELL_A, psi_L, np.where(x < WELL_B, psi_m, psi_R))


def get_training_points(n):
    a, b = DOMAIN
    train_x_1 = np.linspace(a, b, n)
    train_y_1 = np.zeros_like(train_x_1)
    train_x_0 = np.array([0, 1])
    train_y_0 = u_exact(train_x_0)
    train_i_0 = np.ones_like(train_x_0)*0
    train_i_1 = np.ones_like(train_x_1)*1
    train_x = np.concatenate([train_x_0, train_x_1], axis = 0)
    train_i = np.concatenate([train_i_0, train_i_1], axis = 0)
    full_train_x = np.stack([train_x, train_i], axis = -1)
    train_y = np.concatenate([train_y_0, train_y_1], axis = 0)
    return full_train_x, train_y


def get_test_points(n):
    a, b = DOMAIN
    test_x_1 = np.linspace(a, b, n)
    test_y_1 = np.zeros_like(test_x_1)
    test_x_0 = np.array([])#
    test_y_0 = u_exact(test_x_0)
    test_i_0 = np.ones_like(test_x_0)*0
    test_i_1 = np.ones_like(test_x_1)*1
    test_x = np.concatenate([test_x_0, test_x_1], axis = 0)
    test_i = np.concatenate([test_i_0, test_i_1], axis = 0)
    full_test_x = np.stack([test_x, test_i], axis = -1)
    test_y = np.concatenate([test_y_0, test_y_1], axis = 0)
    V = potential(test_x_1)
    print(V.shape)
    return full_test_x, test_y, V, test_x_1
import os




train_x, train_y = get_training_points(10)
test_x, test_y, V, x = get_test_points(100)


data_path = os.path.join(os.path.dirname(__file__),
                        "data.npz")
np.savez_compressed(data_path,
                        train_x = train_x,
                        test_x = test_x,
                        train_y = train_y,
                        test_y = test_y,
                        plot_x = x,
                        potential = V
                    )
print("saved")
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(train_x[:,0], train_y, label = "train")
plt.plot(test_x[:,0], test_y, label = "test")
plt.plot(x, V, label = "V")
plt.legend()

plt.show()


"""
def training_points(n_coll=28, n_obs=7, noise_sd=0.02, seed=2):
    #Return collocation nodes, boundary data, and noisy interior observations.

    #Returns
    #-------
    #Xc : (n_coll,) collocation nodes where L psi = 0 is enforced
    #g  : (n_coll,) PDE right-hand side (all zeros here)
    #Xb : (2 + n_obs,) boundary + observation locations
    #ub : (2 + n_obs,) corresponding values (noisy for interior obs)
    
    rng = np.random.default_rng(seed)
    a, b = DOMAIN

    Xc = np.linspace(a, b, n_coll)
    g = np.zeros(n_coll)

    xb0, ub0 = BC[0]
    xb1, ub1 = BC[1]
    xo = np.sort(rng.uniform(a + 0.06, b - 0.06, n_obs))
    yo = u_exact(xo) + rng.normal(0.0, noise_sd, n_obs)

    Xb = np.concatenate([[xb0, xb1], xo])
    ub = np.concatenate([[ub0, ub1], yo])
    return Xc, g, Xb, ub


if __name__ == "__main__":
    Xc, g, Xb, ub = training_points()
    print(f"Analytical coefficients: B={_B:.6f}, C={_C:.6f}, D={_D:.6f}, E={_E:.6f}, F={_F:.6f}")
    print("\nCollocation nodes Xc:")
    print(np.round(Xc, 4))
    print("\nBoundary + observation locations Xb:")
    print(np.round(Xb, 6))
    print("\nCorresponding values ub:")
    print(np.round(ub, 6))
    xtest = np.array([0.0, 0.35, 0.5, 0.65, 1.0])
    print("\nExact solution at selected points:")
    for xi in xtest:
        print(f"  psi({xi}) = {u_exact(xi):.6f}")
"""
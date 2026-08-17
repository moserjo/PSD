import numpy as np
import os

KAPPA_L = 1.0
KAPPA_R = 6.0
X0 = 0.5
DOMAIN = (0.0, 1.0)
BC = ((0.0, 0.0), (1.0, 1.0))
SOURCE = 0.0

FLUX_CONST = 2.0 / (1.0 / KAPPA_L + 1.0 / KAPPA_R)   # C = 12/7
U_MID = FLUX_CONST * X0 / KAPPA_L                       # u(0.5) = 6/7


def kappa(x):
    return 1+x**2

def u_exact(x):
    x = np.asarray(x, float)
    return 4/np.pi*(np.arctan(x))


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
    test_x_0 = test_x_1#np.array([])#
    test_y_0 = u_exact(test_x_0)
    test_i_0 = np.ones_like(test_x_0)*0
    test_i_1 = np.ones_like(test_x_1)*1
    test_x = np.concatenate([test_x_0, test_x_1], axis = 0)
    test_i = np.concatenate([test_i_0, test_i_1], axis = 0)
    full_test_x = np.stack([test_x, test_i], axis = -1)
    test_y = np.concatenate([test_y_0, test_y_1], axis = 0)
    V = kappa(test_x_1)
    print(V.shape)
    return full_test_x, test_y, V, test_x_1


train_x, train_y = get_training_points(10)
test_x, test_y, V, x = get_test_points(100)


data_path = os.path.join(os.path.dirname(__file__),
                        "data.npz")
np.savez_compressed(data_path,
                        train_x = train_x,
                        test_x = test_x,
                        train_y = train_y,
                        plot_x = x,
                        potential = V
                    )
print("saved")

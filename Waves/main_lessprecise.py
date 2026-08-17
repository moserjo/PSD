
import torch
import gpytorch
import Waves as ex
torch.set_default_dtype(torch.float32)
import os
import numpy as np

data_path = os.path.join(
        os.path.dirname(__file__),
       "data.npz")

data = np.load(data_path)
train_x = torch.tensor(data["train_x"])
test_x = torch.tensor(data["test_x"])
train_y = torch.tensor(data["train_y"])

num_tasks = 2
noise_tensor = torch.tensor(0*np.ones(train_y.shape[0]))


for l in [0.1, 0.5, 1., 3, 10]:
    modified_parameters = { 
                    "amplitude": [1., False, gpytorch.constraints.Positive()], 
                    "lengthscale": [l, False, gpytorch.constraints.GreaterThan(0.1)]}

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise = noise_tensor)                                  
    model = ex.PCGP_Model(train_x, train_y, likelihood, modified_parameters)
    model.eval()
    likelihood.eval()
    with torch.no_grad(),gpytorch.settings.prior_mode(True):
            predictions = model(test_x)
            mean = predictions.mean
            lower, upper = predictions.confidence_region()


    with torch.no_grad():
        K = predictions.covariance_matrix
        eigvals = torch.linalg.eigvalsh(K)

    output_path = os.path.join(
        os.path.dirname(__file__),"results",
        "result_l%.1f_lp.npz"%l)

    with torch.no_grad():   
        np.savez_compressed(
                output_path,
                K = K.numpy(),
                eigvals = eigvals.numpy()[::-1],
                coeff = data["potential"],
                x = data["plot_x"]
            )   
print("saved")

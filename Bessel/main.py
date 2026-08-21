
import torch
import os
import gpytorch
import Bessel as ex
import Vanilla as van
import scipy 
import numpy as np
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from training import train
torch.set_default_dtype(torch.float64)


n = 1
N = 100

num_tasks = 2
end = 5
plot_x = torch.linspace(0, end, N)
#X = torch.cat([test_x+end*i for i in range(num_tasks)], dim = 0)


#training and test data
train_x_0 = torch.tensor([0, 5])
train_x_1 = torch.linspace(0, 5, 20)
train_x = torch.cat([train_x_0, train_x_1], dim = 0)
train_i = torch.cat([torch.zeros_like(train_x_0), torch.ones_like(train_x_1)], dim = 0)
full_train_x = torch.stack([train_x, train_i], dim = -1)
train_y_0 = torch.tensor([scipy.special.jv(n, 0), scipy.special.jv(n, 5)]) 
train_y_1 = torch.zeros_like(train_x_1)
train_y = torch.cat([train_y_0, train_y_1], dim = 0)
test_x = torch.cat([plot_x, plot_x], dim = 0)
test_i = torch.cat([torch.zeros_like(plot_x), torch.ones_like(plot_x)], dim = 0)
full_test_x = torch.stack([test_x, test_i], dim = -1)
test_y_0 = scipy.special.jv(n, plot_x)
test_y_1 = torch.zeros_like(plot_x)
test_y = torch.cat([test_y_0, test_y_0], dim = 0)
print(train_y, "trainy")
#Vanilla just dummy calculation to access kernel similar to subsequent ones
likelihood_vanilla = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-8, 1e-6 ))                                                                                                                       
model_vanilla = van.PCGP_Model(torch.stack([train_x_1, torch.zeros_like(train_x_1)], dim = -1), train_y[2:], likelihood_vanilla, num_tasks = 1, priors=None)
model_vanilla.eval()
likelihood_vanilla.eval()
with torch.no_grad(), gpytorch.settings.prior_mode(True):
    vanilla = model_vanilla(full_test_x[full_test_x[:,-1] == 0])
    mean = vanilla.mean
    lower, upper = vanilla.confidence_region()

K = vanilla.covariance_matrix
log_K = torch.log(abs(K))
data_path = os.path.join(os.path.dirname(__file__), "results", "vanilla.npz")
np.savez_compressed(data_path,
                        K = log_K.numpy(),
                        mean = mean.numpy(),
                        lower = lower.numpy(),
                        upper = upper.numpy(),
                        plot_x = plot_x.numpy())
print("vanilla saved")


#Prior PIGP
parameters = {  "n": [n, False, False],
                "amplitude":[1., True, gpytorch.constraints.Positive()], 
                "lengthscale": [1., True, gpytorch.constraints.Positive()],}
noise_tensor = 1e-6*torch.ones_like(train_y)
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise_tensor)#noise_constraint=gpytorch.constraints.Interval(1e-10, 1e-6))                                                                                                                       
model = ex.PCGP_Model(full_train_x, train_y, likelihood, parameters, num_tasks = num_tasks, priors=None)

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.prior_mode(True):
    prior = model(full_test_x)
    mean = prior.mean
    lower, upper = prior.confidence_region()

K = prior.covariance_matrix
log_K = torch.log(abs(K))

gap = 5  # size of white gap between task blocks (pixels)
size = num_tasks * N + (num_tasks - 1) * gap
log_K_gap = torch.full((size, size), torch.nan)

for i in range(num_tasks):
    for j in range(num_tasks):
        r0 = i * (N + gap)
        c0 = j * (N + gap)
        log_K_gap[r0:r0+N, c0:c0+N] = log_K[i*N:(i+1)*N, j*N:(j+1)*N]

data_path = os.path.join(os.path.dirname(__file__), "results", "prior.npz")
np.savez_compressed(data_path,
                        K = log_K_gap.numpy(),
                        mean = torch.stack([mean[test_i==0], mean[test_i==1]], dim = -1).numpy(),
                        lower = torch.stack([lower[test_i==0], lower[test_i==1]], dim = -1).numpy(),
                        upper = torch.stack([upper[test_i==0], upper[test_i==1]], dim = -1).numpy(),
                        plot_x = plot_x.numpy())
print("prior saved")

#Posterior
model.train()
likelihood.train()
training_iter = 500
with gpytorch.settings.max_cholesky_size(float('inf')):
    train_output = train(model,
                         likelihood,
                         parameters,
                         full_train_x,
                         train_y,
                         training_iter=training_iter,
                         learning_rate = 0.1,
                         laplace = False)
likelihood.eval()
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    posterior = model(full_test_x)
    mean = posterior.mean
    lower, upper = posterior.confidence_region()


K = posterior.covariance_matrix
log_K = torch.log(abs(K))
log_K_gap = torch.full((size, size), torch.nan)

for i in range(num_tasks):
    for j in range(num_tasks):
        r0 = i * (N + gap)
        c0 = j * (N + gap)
        log_K_gap[r0:r0+N, c0:c0+N] = log_K[i*N:(i+1)*N, j*N:(j+1)*N]
data_path = os.path.join(os.path.dirname(__file__), "results", "posterior.npz")
print(train_y_0)
np.savez_compressed(data_path,
                        K = log_K_gap.numpy(),
                        mean = torch.stack([mean[test_i==0], mean[test_i==1]], dim = -1).numpy(),
                        lower = torch.stack([lower[test_i==0], lower[test_i==1]], dim = -1).numpy(),
                        upper = torch.stack([upper[test_i==0], upper[test_i==1]], dim = -1).numpy(),
                        train_x_0 = train_x_0.numpy(),
                        train_y_0 = train_y_0.numpy(),
                        train_x_1 = train_x_1.numpy(),
                        train_y_1 = train_y_1.numpy(),
                        test_y_0 = test_y_0.numpy(),
                    
                        test_y_1 = test_y_1.numpy(),
                        plot_x = plot_x.numpy())
print("posterior saved")



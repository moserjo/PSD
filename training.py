import torch
import gpytorch
from PCGP import laplace_approx, LaplaceResult
from dataclasses import dataclass
import copy
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class train_output:
    parameters_during_training:dict
    laplace:LaplaceResult
    loss:list


def train(model, likelihood, parameters, train_x, train_y, training_iter=50, learning_rate = 0.1, laplace = True):
    parameters_during_training = {key:[] for key in parameters}
    loss_landscape = []

    optimizer = torch.optim.Adam(model.named_parameters(), lr=learning_rate)
    marginal_log_likelihood  = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -marginal_log_likelihood(output, train_y)
            
            for key in parameters:
                    parameters_during_training[key].append(copy.deepcopy(model.covar_module.get_param(key).detach()))
            loss_landscape.append(loss.detach())
            
            if i%100==0 and not device=="cuda":
                print("iteration: ", i, "loss:", loss.item())
            loss.backward(retain_graph = True)
            optimizer.step()

    model.train()
    likelihood.train()        
    optimizer.zero_grad()
    output = model(train_x)  
    final_loss = -marginal_log_likelihood(output, train_y)*train_y.shape[0] #necessary to get correct gradients
    print("Final loss: ", final_loss.item())

    laplace_return = None
    if laplace:
        laplace_return = laplace_approx(parameters, model, final_loss)
    
    loss_landscape = torch.stack(loss_landscape).detach().cpu().numpy() 
    for key in parameters_during_training:
        parameters_during_training[key] =  torch.stack(parameters_during_training[key]).detach().cpu().numpy()
    return train_output(parameters_during_training=parameters_during_training,
                        laplace = laplace_return,
                        loss=loss_landscape,
                  )

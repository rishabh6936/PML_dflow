import time
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib.animation as animation
from model_classes import MLP, Swish, WrappedModel
import circle_losses



#loads the uniformly sampled checkerboard points
def load_checkerboard(batch_size: int = 200, device : str = 'cpu'):
    "Loads the checkerboard data points "
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_temp = torch.rand(batch_size, device=device) - torch.randint(high = 2, size=(batch_size,), device=device) * 2
    x2 = x2_temp + (torch.floor(x1) % 2)
    points = 1.0 * torch.cat([x1[:,None], x2[:,None]], dim = 1 ) / 0.45  #multiple by float number aswell to enforce error checking
    return points.float()


def fm_training(steps: int = 5000, batch_size: int = 512, device: str = 'cpu'):
    "Trains the fm model to learn the checkerboard distribution"
    model = MLP().to(device=device)
    wrapped_model = WrappedModel(model)
    print("Training Flow Matching on Checkerboard...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_time = time.time()
    
    wrapped_model.train()
    for iter in range(steps):
        optimizer.zero_grad()
        x1 = load_checkerboard(batch_size, device=device)
        x0 = torch.rand_like(x1)

        t = torch.rand(batch_size,1, device=device)

        # No need for complex ODE solvers here
        x_t = x0 * (1-t) + t*x1
        u_t = x1 - x0

        v_t = wrapped_model(x_t, t.flatten())
        loss = torch.pow(v_t - u_t, 2)
        loss_mean = torch.mean(loss)
        loss_mean.backward()
        optimizer.step()

        if iter % 500 == 0:
            print ("Training in progress, Iter:", iter, "Loss", loss_mean)


    print(f"Training completed in {time.time() - start_time:.2}s")
    return wrapped_model


def dflow(trained_model, loss_func, steps: int = 20, batch_size: int = 4096, device: str = 'cpu', optmz_steps: int = 600):
    "Optimizes the input image x0 for minimizing the cicle loss and ultimately producing circle as an output"
    wrapped_model = trained_model
    print("Training Flow Matching on Checkerboard with Dflow for Circle...")
    x0 = torch.randn(batch_size, 2, device=device)
    #x0 = torch.rand(batch_size, 2, device=device)
    x0.requires_grad = True
    optimizer = torch.optim.Adam([x0], lr=0.1)
    dt = 1/steps
    wrapped_model.eval()
    history_x0 = [] 
    history_x1 = [] 
    for iter in range(optmz_steps):
        optimizer.zero_grad()
        
        x_t = x0
        for i in range(steps):
            t_inst = i/steps
            t_input = torch.full((batch_size,), t_inst, device=device)
            v_t = wrapped_model(x_t, t_input)
            x_t = v_t * dt + x_t

        if iter % 2 == 0: 
            history_x0.append(x0.detach().cpu().numpy().copy()) 
            history_x1.append(x_t.detach().cpu().numpy().copy())     
        
    
    
        circle_loss = loss_func(x_t)
        circle_loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print(f"Iter {iter:4d}: Circle Loss {circle_loss.item():.6f}")


    return history_x0, history_x1






    


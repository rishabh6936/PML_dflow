import time
import torch
import torch.nn as nn
from torch import Tensor

def donut_loss(x: Tensor) -> Tensor:
    "makes a donut shaped circle"
    radius = torch.norm(x, dim=1)
    # Penalize only if radius < 1.8 or radius > 2.2
    too_small = torch.relu(1.8 - radius)
    too_big = torch.relu(radius - 2.2)
    return torch.mean(too_small**2 + too_big**2)

def calc_circle_loss(x : Tensor) -> Tensor:
    "standard circle of radius 2.0"
    norm = torch.norm(x , dim = 1)  # radius
    target_radius = 2.0
    loss = torch.pow(norm - target_radius,2)
    mean_loss = torch.mean(loss)                 #taking the mean loss for the whole batch
    return mean_loss

def crescent_loss(x: Tensor) -> Tensor:
    "moon like structure"
    #Circle Loss
    radius = torch.norm(x, dim=1)
    rad_loss = torch.mean((radius - 2.0)**2)

    #Angle Loss
    
    angle = torch.atan2(x[:, 1], x[:, 0])
    safe_zone = 1.25
    angle_violation = torch.relu(torch.abs(angle) - safe_zone)
    
    angle_loss = torch.mean(angle_violation**2) 

    return rad_loss + 2.0 * angle_loss

def repulsion_loss(x: Tensor) -> Tensor:
    "circle where the points are evently distributed on the circle circumference"
    # first objective
    radius = torch.norm(x, dim=1)
    circle_error = torch.mean((radius - 2.0)**2)
    

    pdist = torch.pdist(x)
    
    #next objective
    repulsion = torch.mean(torch.exp(-pdist * 10.0)) 
    total_loss = circle_error + 0.05 * repulsion
    
    return total_loss
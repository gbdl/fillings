import numpy as np
import torch
from torch.func import vmap, vjp, jacrev

  
#returns the trace of a 2 tensor whose both indices are down
def traceDown(g, tensor):
    TUd = torch.linalg.solve(g, tensor)
    return torch.trace(TUd)
 

#I want the geometric functions to take as input the christoffels and the metric at a given point, rather than computing it

#Ricci with two lower indices.
#diff_christ has the derivative index to be the last one
def ricci(christ, diff_christ):
    c = christ
    dc = diff_christ
    return torch.einsum("ijki",dc)-(torch.einsum("ikil",dc)).permute(1,0)+torch.einsum("iip,pjk->jk", c, c)-torch.einsum("ijp,pik->jk", c, c)

def ricci_scalar(g, christ, diff_christ):
    ricci_tensor = ricci(christ, diff_christ)
    return traceDown(g, ricci_tensor)


## metric has to be a function here
def compute_christoffel(metric, x):
    n = len(x)
    diff_metric = jacrev(metric, argnums=0) #Done in this way it puts the derivative index as last one.
    g = metric(x)
    dg = diff_metric(x)
    dg_ordered = dg.permute(2,0,1) #This is to put the derivative index as first


    #Since linalg.solve works when the RHS B is a vector, I put the extra indices in a single nxn index
    christ_1 = 0.5*torch.linalg.solve(g,dg.view(n,n*n))
    #Then we convert back to a matrix 
    christ_1 = christ_1.view(n,n,n) 
    christ_1kij = christ_1.permute(0, 2, 1)

    christ_2kij = christ_1

    christ_3 = -0.5*torch.linalg.solve(g, dg_ordered.view(n,n*n) )
    christ_3kij = christ_3.view(3,3,3) 

    return christ_1kij+christ_2kij+christ_3kij

## metric has to be a function here, as well as compute_christoffel
#the derivative index is the last one

def der_christoffel(metric, x):
    diff_christ = jacrev(compute_christoffel, argnums=1) #argnums here specifies that the jacobian is with respect to the second argument of the function compute_christoffel
    return diff_christ(metric, x)


#returns the covariant derivative of a 1 form, which has to be a function
#The derivative index is the first one


def covD_down(form, christ, x):
    diff_eta = jacrev(form) #argnums here specifies that the jacobian is with respect to a certain argument 
    deta = diff_eta(x) #remember that the derivative index is the last one
    eta_k = form(x)
    
    ## The order needs to be double checked!
    # agrees with wiki formula but now with others!

    cdeta_ij = deta-torch.einsum("kij,k->ij", christ, eta_k)

    cdeta_ordered = cdeta_ij.permute(1,0) #This is to put the derivative index     
    return cdeta_ordered

#This projects a two tensor using projector
def project_2_tensor(projector, tensor_2):
    return torch.einsum("am,bn,mn->ab",projector, projector, tensor_2)


## These functions are pointwise, I batch them with vmap when I need it


## This is the hyperbolic metric
def metric_hyp(x):
    z = x[0]
    return torch.eye(3, device = x.device, dtype = torch.float32)/(z*z)

def christoffels_hyp(x):
    z = x[0]
    device = x.device
    return torch.stack([
        torch.stack([-(1/z), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), 1/z, torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), 1/z]),
        torch.stack([torch.tensor(0.0, device = device), -(1/z), torch.tensor(0.0, device = device), -(1/z), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device)]),
        torch.stack([torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), -(1/z), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device), -(1/z), torch.tensor(0.0, device = device), torch.tensor(0.0, device = device)])
    ], dim=0).reshape(3, 3, 3)
 

#Defines the vector field used for gauge fixing
def xi(metric, christ, chris_ref):
    B = christ-chris_ref
    #unsqueeze used to add a batch dimension so that A^{-1} does not contract the first index of B
    return torch.einsum("ijj-> i",torch.linalg.solve(metric.unsqueeze(0), B))

def norm_sq_vec(metric, v):
    return (metric@v)@v


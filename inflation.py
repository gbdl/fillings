from torch.optim.optimizer import Optimizer, required
import numpy as np
import torch

class ECDSep_scaled(Optimizer):
    '''
    Optimizer based on the new separable Hamiltonian and generalized bounces
    lr (float): learning rate, called Delta t in the paper (required)
    F0 (float): expected minimum of the objective
    deltaEn (float): initial energy
    eta (float): hyperparameter that controls the concentration of the measure (required).
                 It has to be >= 1. Increasing it concentrates the measure towards the bottom of the basin, and it is useful for pure optimization problems where the goal to find smallest loss. Tested up to eta = 5.
    consEn (bool): whether the energy is conserved or not
    weight_decay (float): weight decay, implemented as L^2 term
    nu (float): chaos hyperparameter
    s (float): regularization switch
    '''

    def __init__(self, params, lr=required, F0=0., eps1=1e-10, eps2=1e-40, deltaEn=0., nu=1e-5, s=1., weight_decay=0, eta=required, consEn=True):
        defaults = dict(lr=lr, F0=F0, eps1=eps1, eps2=eps2, deltaEn=deltaEn, nu=nu, s=s, weight_decay=weight_decay, eta=eta, consEn=consEn)
        self.F0 = F0
        self.s = s
        if self.s == 1:
            self.deltaEn = deltaEn
        else:
            self.deltaEn = 1
        self.eta = eta
        self.consEn = consEn
        self.iteration = 0
        self.lr = lr
        self.eps1 = eps1
        self.eps2 = eps2
        self.weight_decay = weight_decay
        self.nu = nu
        self.dim = 0
        super(ECDSep_scaled, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # compute q^2 for the L^2 weight decay
        self.q2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)

        if self.weight_decay != 0:
            for group in self.param_groups:
                for q in group["params"]:
                    self.q2.add_(torch.sum(q**2))
                    
        V = (loss + 0.5*self.weight_decay*self.q2 - self.F0)**self.eta

        # Initialization
        if self.iteration == 0:
            
            # Define random number generator and set seed
            self.generator = torch.Generator(device = self.param_groups[0]["params"][0].device)
            self.generator.manual_seed(self.generator.seed())
            
            # Initial value of the loss
            V0 = V
            
            # Initial energy and its exponential
            self.energy = torch.log(V0)+torch.log(torch.tensor(self.s+self.deltaEn))
            self.expenergy = torch.exp(self.energy)

            self.min_loss = float("inf")

            if self.consEn == False:
                self.normalization_coefficient = 1.0
                
            # Initialize the momenta along (minus) the gradient
            p2init = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    self.dim += q.numel()
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    if q.grad is None:
                        continue
                    else:
                        p = -d_q 
                        param_state["momenta"] = p
                    
                    p2init.add_(torch.norm(p)**2)  
            # Normalize the initial momenta such that |p(0)| = deltaEn
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    param_state["momenta"].mul_(torch.sqrt(self.deltaEn/p2init))   

            self.p2 = torch.tensor(self.deltaEn)

            ## Now rescale the hypers
            self.lr = self.lr/np.sqrt(self.eta)
            self.nu = self.nu/np.sqrt(self.dim)


        if V > self.eps2:

            # Scaling factor of the p for energy conservation
            if self.consEn == True:
                p2true = ((self.expenergy / V)-self.s)
                if torch.abs(p2true-self.p2) < self.eps1:
                    self.normalization_coefficient = 1.0
                elif  p2true < 0:
                    self.normalization_coefficient = 1.0
                elif self.p2 == 0:
                    self.normalization_coefficient = 1.0
                else:
                    self.normalization_coefficient = torch.sqrt(p2true / self.p2)

            # Update the p's and compute p^2 that is needed for the q update
            self.p2 = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            for group in self.param_groups:
                for q in group["params"]:
                    param_state = self.state[q]
                    d_q = q.grad.data 
                    p = param_state["momenta"]
                    # Update the p's
                    if q.grad is None:
                        continue
                    else:
                        p.mul_(self.normalization_coefficient)
                        p.add_(- self.lr * (self.eta/(loss + 0.5*self.weight_decay*self.q2 - self.F0))*(d_q+self.weight_decay*q))
                        self.p2.add_(torch.norm(p)**2)
            
            #Update the q's and add tiny rotation of the momenta
            p2new = torch.tensor(0., device = self.param_groups[0]["params"][0].device)
            pnorm = torch.sqrt(self.p2)
            for group in self.param_groups:
                for q in group["params"]:  

                    #Update of the q
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    q.data.add_(self.lr * 2 * param_state["momenta"]/(self.s+self.p2))

                    #Add noise to the momenta
                    z = torch.randn(p.size(), device=p.device, generator = self.generator)
                    param_state["momenta"] = p/pnorm + self.nu*z 
                    p2new += torch.dot(param_state["momenta"].view(-1),param_state["momenta"].view(-1))

            # Normalize new direction
            for group in self.param_groups:
                for q in group["params"]:  
                    param_state = self.state[q]
                    p = param_state["momenta"]
                    param_state["momenta"] = pnorm * p /torch.sqrt(p2new)
                   
            self.p2 = pnorm**2 

            self.iteration += 1

        return loss

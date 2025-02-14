import torch
from torch.func import vmap, vjp, jacrev
from geometry import *


## Here we define a class that computes a metric and its geometric quantities from a NN

class non_diagonal_metric:
    def __init__(self, manifold, models, dtype = torch.float32):
        self.models = models
        self.manifold = manifold
        self.dtype = dtype

    #This function constructs the metric from the functions we use to parametrize it
    #which are not directly the metric components, but they are multiplied by some functions that absorb divergences

    def metric_from_comp(self, x, g_zz, g_x1x1, g_x2x2, g_zx1, g_zx2, g_x1x2 ):
        device = x.device
        z = torch.dot(x,torch.eye(self.manifold.n, device = device, dtype =  self.dtype)[0])
        z2 = z**2
        eps = 1e-8

        return torch.stack(
            (torch.stack( (g_zz(x)/(z2*(self.manifold.rj-z)), g_zx1(x) + eps, g_zx2(x) + eps)),
            torch.stack( (g_zx1(x)+ eps,((self.manifold.rj-z)*g_x1x1(x))/z2, g_x1x2(x) + eps) ),
            torch.stack( (g_zx2(x) + eps,g_x1x2(x) + eps,g_x2x2(x)/z2 )))
            )
        
    def metric(self, x):
        return self.metric_from_comp(x, self.models[0], self.models[1], self.models[2], self.models[4], self.models[5], self.models[3])
    
    def compute_christoffel(self, x):
        return compute_christoffel(self.metric, x)

    def der_christoffel(self, x):
        diff_christ = jacrev(self.compute_christoffel)
        return diff_christ(x)



class approximate_DF:
    def __init__(self, manifold, dtype = torch.float32, gamma = 1., shift = 0):
        self.k = manifold.k
        self.rj = manifold.rj
        self.n = manifold.n    
        self.dtype = dtype
        self.alpha = manifold.L2/manifold.L1
        self.gamma = gamma #interpolating factor
        self.shift = shift #shift for the interpolation

    ## These are used to build the DF initial piecewise metric
    def metric_DF(self, x):
        ## TODO: To improve by writing things in terms of alpha instead of hardcoded L1,L2
        
        #It is not possible to use vmap over if else statements
        #Need to use torch.where
        device = x.device
        #z = x[0]
        z = torch.dot(x,torch.eye(self.n, device = device, dtype =  self.dtype)[0])
        def gzz(z):
            zsq = z**2            
            to_return_if_zll1 = 1/zsq
            to_return_else  = pow(self.rj,2)/(pow(self.rj,2)*pow(z,2) - pow(z,4))
            return torch.where(z<1, to_return_if_zll1, to_return_else)

        def gyy(z):
            zsq = z**2
            to_return_if_zll1 = 1/zsq
            to_return_else = ((-1 + pow(self.rj,2)/pow(z,2))/(4.*(-1 + pow(self.rj,2))) +  pow(self.k,2)/pow(z,2))/(0.25 + pow(self.k,2))
            return torch.where(z<1, to_return_if_zll1, to_return_else)

        def gxx(z):
            zsq = z**2
            to_return_if_zll1 = 1/zsq
            to_return_else = ((pow(self.k,2)*(-1 + pow(self.rj,2)/pow(z,2)))/(-1 + pow(self.rj,2)) +      1/(4.*pow(z,2)))/(0.25 + pow(self.k,2))
            return torch.where(z<1, to_return_if_zll1, to_return_else)

        def gxy(z):
            to_return_if_zll1 = 0*z
            to_return_else = (-2*self.k*(-1 + pow(z,2)))/((1 + 4*pow(self.k,2))*(-1 + pow(self.rj,2))*pow(z,2))
            return torch.where(z<1, to_return_if_zll1, to_return_else)

        return torch.stack(
            (torch.stack( (gzz(z),torch.tensor(0.0, device = device),torch.tensor(0.0, device = device) )),
            torch.stack( (torch.tensor(0.0, device = device),gxx(z),gxy(z) )),
            torch.stack( (torch.tensor(0.0, device = device),gxy(z),gyy(z) )))
            )

    #This defines a manual smooth interpolation
    def metric_DF_interp(self, x):
        device = x.device
        alpha2 = self.alpha**2
        k2 = self.k**2
        z = torch.dot(x,torch.eye(self.n, device = device, dtype =  self.dtype)[0])
        
        def chi(z):
            r = self.rj/z
            #res = torch.tanh((-2./self.rj)*(r- (5./4)*self.rj))
            #res = torch.tanh((-10./self.rj)*(r- (5./4)*self.rj))
            #res = torch.tanh(-((self.gamma/self.rj)**(0.5))*(r- ((5./4.)+self.shift)*self.rj))
            res = torch.tanh(-((2*self.gamma)/self.rj)*(r- ((5./4.)+self.shift)*self.rj))
            return 0.5*(res+1)

        def W(z):
            return (self.rj**2)/(z**2)-chi(z)

        def gzz(z):
            return pow(self.rj,2)/(pow(z,4)*W(z))

        def gyy(z):
            zsq = z**2
            W1 = W(torch.tensor(1.0, device = device))
            to_return = alpha2 * (W(z)/W1) + k2/zsq
            return to_return/(k2+alpha2)

        def gxx(z):
            zsq = z**2
            W1 = W(torch.tensor(1.0, device = device))
            to_return = k2 * (W(z)/W1) + alpha2/zsq
            return to_return/(k2+alpha2)

        def gxy(z):
            zsq = z**2
            W1 = W(torch.tensor(1.0, device = device))
            to_return = self.k*self.alpha * ((W(z)/W1) - 1./zsq)
            return to_return/(k2+alpha2)
     
        return torch.stack(
            (torch.stack( (gzz(z),torch.tensor(0.0, device = device),torch.tensor(0.0, device = device) )),
            torch.stack( (torch.tensor(0.0, device = device),gxx(z),gxy(z) )),
            torch.stack( (torch.tensor(0.0, device = device),gxy(z),gyy(z) )))
            )


    def hyp_metric(self, x):
        device = x.device
        z = torch.dot(x,torch.eye(self.n, device = device, dtype =  self.dtype)[0])
        zsqinv = z**(-2.)

        return torch.stack(
            (torch.stack( (zsqinv,torch.tensor(0.0, device = device),torch.tensor(0.0, device = device) )),
            torch.stack( (torch.tensor(0.0, device = device),zsqinv,torch.tensor(0.0, device = device) )),
            torch.stack( (torch.tensor(0.0, device = device),torch.tensor(0.0, device = device),zsqinv )))
            )




    def compute_christoffel_interp(self, x):
        return compute_christoffel(self.metric_DF_interp, x)

    def der_christoffel_interp(self, x):
        diff_christ = jacrev(self.compute_christoffel_interp)
        return diff_christ(x)

    def compute_christoffel_hyp(self, x):
        return compute_christoffel(self.hyp_metric, x)

    def der_christoffel_hyp(self, x):
        diff_christ = jacrev(self.compute_christoffel_hyp)
        return diff_christ(x)





            



    


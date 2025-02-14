import numpy as np
import torch
from torch import nn
from torch.func import vmap, vjp, jacrev


##Relative error between two tensor
##Useful because we might have points with large values (e.g. close to z = zero)

class RelativeDiff(torch.nn.Module):
    def __init__(self, epsilon = 1e-12):
        super().__init__()
        self.epsilon = epsilon

    def relative_diff(self, A, B):
        abs_A = torch.abs(A)
        abs_B = torch.abs(B)
    
        # Compute relative difference
        relative_diff = (A - B) / (abs_A + abs_B + self.epsilon) #(B, 3, 3) tensor
        
        return relative_diff


class MeanRelativeError(RelativeDiff):
    def __init__(self, norm_type = "L2", epsilon = 1e-12):
        super().__init__(epsilon)
        if norm_type == "L1":
            self.criterion = self.L1_norm
        else:
            self.criterion = self.L2_norm

    def L2_norm(self, tensor):
        #same as (torch.sum(tensor**2)**0.5)/tensor.numel()
        return torch.linalg.norm(tensor)/tensor.numel()

    def L1_norm(self, tensor):
        #same as (torch.sum(torch.abs(tensor)))/tensor.numel()
        return torch.linalg.norm(tensor.flatten(), ord = 1)/tensor.numel()


    def forward(self, A, B):
        relative_diff_tensor = self.relative_diff(A, B)
        return self.criterion(relative_diff_tensor)

    def return_ordered_points_bry(self, A, B, points_A, points_B):
        #Returns points ordered by the relative_diff error
        relative_diff_tensor = self.relative_diff(A,B) #(batch, 2, 2) tensor
        points_A_new = points_A.detach().clone()
        points_B_new = points_B.detach().clone()
        
        # I first take the maximum over the 2x2 dimensions
        #can't use view because the tensor is not contiguous in memory
        relative_diff_flat = relative_diff_tensor.reshape((len(points_A_new),-1)) #(batch, 4)
        relative_diff_maxs = torch.max(relative_diff_flat,1).values
        indices_ordered = torch.sort(relative_diff_maxs, descending = True).indices

        return points_A_new[indices_ordered], points_B_new[indices_ordered]

        
    
    def return_ordered_points(self, A, B, points):
        #Returns points ordered by the relative_diff error
        relative_diff_tensor = self.relative_diff(A,B) #(batch, 3, 3) tensor

        points_new = points.detach().clone()
        # I first take the maximum over the 3x3 metric
        relative_diff_flat = relative_diff_tensor.view((len(points_new),-1)) #(batch, 9)
        relative_diff_maxs = torch.max(relative_diff_flat,1).values
        indices_ordered = torch.sort(relative_diff_maxs, descending = True).indices

        #return the points ordered by max error in any of the 3x3 components   
        return points_new[indices_ordered]


## I also want to track the maximum relative error 
class MaxRelativeError(RelativeDiff):

    def __init__(self, epsilon=1e-12):
        super().__init__(epsilon)

    def forward(self, A, B):
        # Compute relative difference 
        relative_diff_tensor = self.relative_diff(A, B) #(B, 3, 3) tensor
        return torch.max(torch.abs(relative_diff_tensor))







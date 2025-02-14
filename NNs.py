import torch
import torch.nn as nn

class Res_FC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_first_layer = nn.Linear(config.input_dim, config.width )


        #Putting a layer norm before every linear layer, except the initial one.
        
        self.fc_inner_layers = nn.ModuleList([nn.Linear(config.width, config.width) for _ in range(config.depth)])
        self.fc_last_layer = nn.Linear(config.width, config.output_dim)
        
        if config.activation == "gelu": self.activation = nn.GELU()
        elif config.activation == "tanh": self.activation = nn.Tanh()
        else: self.activation = nn.Identity()

        self.ln_inner = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(config.depth)])
        self.ln_last = nn.LayerNorm(config.width)

    def forward(self, x):

        
        x = self.fc_first_layer(x)
    
        for i in range(len(self.fc_inner_layers)):
            x = x + self.activation(self.fc_inner_layers[i](self.ln_inner[i](x)))
        
        x = self.fc_last_layer(self.ln_last(x)) #(B,outpunt_dim)

        return x.squeeze() #if the output is (B,1) just returns (B)

#The same as Res_FC, but it initializes the last layer to zero
class Res_FC_zero(Res_FC):
    def __init__(self, config):
        super().__init__(config) #Calls the constructor of the parent class
        
        nn.init.normal_(self.fc_last_layer.weight, mean = 0.0, std = 1e-8)
        nn.init.normal_(self.fc_last_layer.bias, mean = 0.0, std = 1e-10)


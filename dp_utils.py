# dp_utils.py
import torch
import syft as sy

# Hook into PyTorch
sy.TorchHook(torch)

def apply_differential_privacy(model, noise_multiplier=1.0, max_grad_norm=1.0, apply_to=None):
    for param in model.parameters():
        #may need to adjust clamp limits pending results
        param.grad = torch.clamp(param.grad, max=max_grad_norm, min=-max_grad_norm)
        param.grad += torch.distributions.laplace.Laplace(0, noise_multiplier).sample(param.grad.size())

    # apply_to allows for small numbers of specified parameters/gradients to have DP applied
    # setting the value to None effectively deactivates DP, which may remove the need to have an activation flag
    # setting apply_to to ALL parameters requires all params added to a list; recommend using the code above instead
    # if apply_to is None:
    #     apply_to = []
    
    # for name, param in model.named_parameters():
    #     if name in apply_to:  # Apply DP only to specified parameters
    #         param.grad = torch.clamp(param.grad, max=max_grad_norm, min=-max_grad_norm)
    #         param.grad += torch.distributions.laplace.Laplace(0, noise_multiplier).sample(param.grad.size())

### Other code we may need in the training model ###    
#Inside the class initialization call for a flower library FL client
#def __init__(self, model, trainloader, testloader, dp_enabled=False, dp_layers=None):
    # self.dp_enabled = dp_enabled
    # self.dp_layers = dp_layers if dp_layers is not None else []
# In the training function, at bottom
    # if self.dp_enabled:
    #     apply_differential_privacy(self.model, apply_to=self.dp_layers)
    # self.optimizer.step()
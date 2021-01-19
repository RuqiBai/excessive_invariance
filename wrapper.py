from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch
import torch.nn.functional as F


class ModelWrapper(nn.Module):
    """
    Wrapping the model to fit the requirement of the ART toolbox.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, ori=False):
        out, out_bij, c, d  = self.model.forward(x)
        if ori:
            return out, out_bij, c, d
        else:
            return out

 

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = layer.SeqToANNContainer(
            nn.Linear(in_features, num_classes)
            )
        self.sn = neuron.IFNode(step_mode='m', detach_reset=True)

    def forward(self, features):
        # feature.shape: nsteps, batch_size, in_features
        x = self.fc(features)
        x = self.sn(x)
        return x.mean(dim=0)  # batch_size, num_classes
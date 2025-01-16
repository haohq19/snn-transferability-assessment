import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

__all__ = ['SpikingMLP', 'spiking_mlp5', 'spiking_mlp2', 'spiking_mlp3']


class SpikingMLP(nn.Module):

    def __init__(self, num_layers, in_features=224*224*2, num_classes=1000, T=8):
        super(SpikingMLP, self).__init__()
        self.T = T
        self.num_layers = num_layers
        
        self.avgpool1 = layer.SeqToANNContainer(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.sn1 = neuron.IFNode(step_mode='m', detach_reset=True)
        in_features = in_features // 4
        self.layer0 = self._make_layer(in_features, 512)
        for i in range(1, num_layers-1):
            setattr(self, f'layer{i}', self._make_layer(512, 512))

        self.output =layer.SeqToANNContainer(
            nn.Linear(512, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, in_features, out_features):
        fc = layer.SeqToANNContainer(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
        )   
        sn = neuron.IFNode(step_mode='m', detach_reset=True)
        mlp_layer = nn.Sequential(fc, sn)
        return mlp_layer

    def _forward_impl(self, x):
        # x.shape = (T, B, C, H, W)
        x = self.sn1(self.avgpool1(x))
        x = x.view(x.size(0), x.size(1), -1)
        for i in range(self.num_layers - 1):
            x = getattr(self, f'layer{i}')(x)
        x = self.output(x)
        return x.mean(dim=0)    # x.shape = (B, C)

    def forward(self, x):
        return self._forward_impl(x)


def _spiking_mlp(num_layers, **kwargs):
    model = SpikingMLP(num_layers, **kwargs)
    return model


def spiking_mlp5(**kwargs):
    return _spiking_mlp(5, **kwargs)


def spiking_mlp2(**kwargs):
    return _spiking_mlp(2, **kwargs)


def spiking_mlp3(**kwargs):
    return _spiking_mlp(3, **kwargs)





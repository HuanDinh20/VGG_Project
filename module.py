from torch import nn
import torch


class VGG(nn.Module):
    def __init__(self, architecture, num_classes):
        super(VGG, self).__init__()
        vgg_blocks = []
        for num_conv, (in_channels, out_channels) in architecture:
            vgg_blocks.append(vgg_block(num_conv, in_channels, out_channels))
        self.net = nn.Sequential(
            *vgg_blocks,
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(num_classes))

    def forward(self, X):
        X = self.net(X)
        return X

    def layer_summary(self, X_shape):
        X = torch.rand(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape: ", X.shape)

    @staticmethod
    def xavier_uniform(layer):
        if type(layer) in [nn.Conv2d, nn.Linear]:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.00001)


def vgg_block(num_conv, num_InChannels, num_OutChannels):
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(num_InChannels, num_OutChannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

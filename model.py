import torch
import torch.nn as nn
import numpy as np
from itertools import chain
from torch.autograd import Variable
import torch.autograd as autograd

class Condition(nn.Module):
    def __init__(self, instance_dim, labels_dim):
        super(Condition, self).__init__()
        self.gammas = nn.Embedding(labels_dim, instance_dim)
        self.betas = nn.Embedding(labels_dim, instance_dim)

    def forward(self, batch, labels):
        gamma = self.gammas(labels).unsqueeze(2).unsqueeze(3).expand(batch.shape)
        beta = self.betas(labels).unsqueeze(2).unsqueeze(3).expand(batch.shape)
        return batch * gamma + beta


class BNGenerator(nn.Module):
    '''
    Generates (img_size, img_size) image.
    '''

    def __init__(self, img_size, z_dim, y_dim, n_channels, n_features):
        '''
        :param img_size: image width (height)
        :param z_dim: dimensionality of a latent vector
        :param y_dim: dimensionality of a label vector
        :param n_channels: number of image channels
        :param n_features: number of features for the last layer
        '''

        super(BNGenerator, self).__init__()

        assert np.log2(img_size).is_integer(), 'Image size must be a power of 2'

        n_in = n_features // 2
        current_img_size = 4
        while current_img_size != img_size:
            current_img_size *= 2
            n_in *= 2

        layers = []

        layers.append(nn.ConvTranspose2d(z_dim, n_in, 4))
        layers.append(nn.BatchNorm2d(n_in, affine=False))
        layers.append(Condition(n_in, y_dim))
        layers.append(nn.ReLU())

        current_img_size = 4
        while current_img_size != img_size // 2:
            layers.append(nn.ConvTranspose2d(n_in, n_in // 2, 4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(n_in // 2, affine=False))
            layers.append(Condition(n_in // 2, y_dim))
            layers.append(nn.ReLU())

            n_in //= 2
            current_img_size *= 2

        layers.append(nn.Conv2d(n_in, n_in, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_in, affine=False))
        layers.append(Condition(n_in, y_dim))
        layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(n_features, n_channels, 4, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.layers = layers

    def forward(self, z, y):
        result = z.unsqueeze(2).unsqueeze(3)

        for layer in self.layers:
            classname = layer.__class__.__name__
            if classname.find('Condition') != -1:
                result = layer(result, y)
            else:
                result = layer(result)

        return result

    def __str__(self):
        return '\n'.join(('Layer {}: {}'.format(i, layer) for i, layer in enumerate(self.layers)))

    def parameters(self):
        all_layers_params = [layer.parameters() for layer in self.layers]
        return chain(*all_layers_params)

    def cuda(self, **kwargs):
        for layer in self.layers:
            layer.cuda(**kwargs)


class ACDiscriminator(nn.Module):
    '''
    Decides if (img_size, img_size) image is fake or not.
    '''

    def __init__(self, img_size, n_classes, n_channels, n_features):
        '''
        :param img_size: image width (height)
        :param y_dim: dimensionality of a label vector
        :param n_channels: number of channels
        :param n_features: number of features for the first layer
        '''

        super(ACDiscriminator, self).__init__()

        assert np.log2(img_size).is_integer(), 'Image size must be a power of 2'
        self.img_size = img_size

        image_features = nn.Sequential()
        image_features.add_module('C-{}-{}'.format(n_channels, n_features),
                               nn.Conv2d(n_channels, n_features, 4, stride=2, padding=1))
        image_features.add_module('ReLU-{}'.format(n_features), nn.ReLU())

        current_size = img_size // 2
        n_in = n_features
        while current_size != 4:
            image_features.add_module('C-{}-{}'.format(n_in, 2 * n_in),
                                   nn.Conv2d(n_in, 2 * n_in, 4, stride=2, padding=1))
            # image_features.add_module('BNorm-{}'.format(2 * n_in), nn.BatchNorm2d(2 * n_in))
            image_features.add_module('ReLU-{}'.format(2 * n_in), nn.ReLU())

            n_in *= 2
            current_size //= 2

        wasserstein = nn.Conv2d(n_in, 1, 4)
        logits = nn.Linear(n_in * current_size * current_size, n_classes)

        self.image_features = image_features
        self.wasserstein = wasserstein
        self.logits = logits

    def forward(self, x):
        image_features = self.image_features(x)
        wasserstein = self.wasserstein(image_features)
        logits = self.logits(image_features.view(x.size(0), -1))
        return wasserstein, logits

    def grad_norm(self, x, grad_outputs):
        x_var = Variable(x, requires_grad=True)

        output, _ = self.forward(x_var)

        grad = autograd.grad(output, x_var, grad_outputs=grad_outputs, only_inputs=True,
                             retain_graph=True, create_graph=True)[0]

        x_grad = grad.contiguous().view(grad.size(0), -1)
        return x_grad.norm(2, dim=1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

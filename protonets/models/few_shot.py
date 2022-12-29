import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchinfo import summary

from torch.autograd import Variable

from protonets.models import register_model

from .utils import *

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Protonet(nn.Module):
    def __init__(self, encoder, projector=None, dist='euclidean'):
        super(Protonet, self).__init__()
        
        self.encoder = encoder
        self.projector = projector
        self.dist = dist

        # summary(encoder, (1,28,28))

    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        # print('1', z[:n_class*n_support].shape)
        # print('2', z[:n_class*n_support].view(n_class, n_support, z_dim).shape)
        # print('3', z[n_class*n_support:].shape)
        # assert False

        if self.projector:
            z_proto = self.projector.forward(z[:n_class*n_support].view(n_class, n_support, z_dim))
            # z_proto = z_proto.view(n_class, z_dim)
        else:
            z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        # print('2', z_proto.shape)

        if self.dist == 'euclidean':
            dists = euclidean_dist(zq, z_proto)
        elif self.dist == 'cosine':
            dists = cosine_dist(zq, z_proto)
        elif self.dist == 'l1':
            dists = l1_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    print(kwargs)
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']
    dist = kwargs['dist']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def conv_block_proj(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    # summary(encoder, x_dim)

    projector = nn.Sequential(
        conv_block_proj(5, hid_dim),
        conv_block_proj(hid_dim, hid_dim),
        conv_block_proj(hid_dim, hid_dim),
        conv_block_proj(hid_dim, 1),
        Flatten()
    )

    return Protonet(encoder, None, dist)

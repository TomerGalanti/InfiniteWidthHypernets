from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Hypernet(nn.Module):
    def __init__(self, args, input_dim1, input_dim2, depth_f, hidden_f, depth_g, hidden_g, bottleneck_dim=100, output_dim=1):
        super(Hypernet, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_f = hidden_f
        self.hidden_g = hidden_g
        self.depth_f = depth_f
        self.depth_g = depth_g
        self.args = args
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim


        self.netF = nn.Sequential(
            nn.Linear(input_dim1, hidden_f),
            nn.ReLU(),
        )

        for i in range(depth_f-2):
            self.netF = nn.Sequential(
                self.netF,
                nn.Linear(hidden_f, hidden_f),
                nn.ReLU(),
            )

            if i == depth_f-3:
                self.netF = nn.Sequential(
                    self.netF,
                    nn.Linear(hidden_f, bottleneck_dim),
                    nn.ReLU(),
                )
        self.netF = nn.Sequential(
            self.netF,
            nn.Linear(bottleneck_dim, input_dim2 * hidden_g +
                      (depth_g-2)*hidden_g**2 + hidden_g * self.output_dim),
        )

    def forward(self, x, y):

        x = x.view(-1,self.input_dim1).float()
        y = y.view(-1,self.input_dim2).float()

        if self.args.task == 'image rep':
            y = y

        weights = self.netF(x)
        weights = weights.unsqueeze(1)
        weights_layer1 = weights[:,:,:self.input_dim2*self.hidden_g]\
            .view(-1, self.hidden_g, self.input_dim2)
        output = nn.ReLU()(torch.bmm(weights_layer1, y.view(-1, y.shape[1], 1)))

        l = self.input_dim2 * self.hidden_g

        for i in range(self.depth_g-2):
            weights_layer = weights[:, :,self.hidden_g**2*i+l:self.hidden_g**2*(i+1)+l] \
                .view(-1, self.hidden_g, self.hidden_g)
            output = torch.bmm(weights_layer, output)# \
                #.view(-1, self.hidden_g)

        if self.depth_g == 2:
            weights_layer2 = weights[:, :, l:] \
                .view(-1, self.output_dim, self.hidden_g)
        else:
            weights_layer2 = weights[:,:,self.hidden_g**2*(i+1)+l:]\
            .view(-1, self.output_dim, self.hidden_g)

        #output = nn.ReLU()(torch.bmm(weights_layer1, y.view(-1,y.shape[1],1)))
        output = torch.bmm(weights_layer2, output)\
            .view(-1, self.output_dim)

        # if self.args.task == 'rotations':
        #     output = F.log_softmax(output, dim=1)
        # elif self.args.task == 'pixels':
        #     output = output

        return output

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_cf(nn.Module):

    def __init__(self):
        super(Net_cf, self).__init__()
        # 1 input image channel, 6 output channels, 2 square convolution
        # kernel
        self.layer1 = nn.Conv2d(1, 6, 3)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.layer2 = nn.Linear(912, 64)  # from image dimension
        self.readout = nn.Linear(64, 1)

    def forward(self, x):
        # pdb.set_trace()
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.layer1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        # pdb.set_trace()
        # flatten all dimensions except the batch dimension
        x = F.relu(self.layer2(x))
        x = self.readout(x)
        return x


class Net_fc(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_fc, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1,
                                  out_channels=64,
                                  kernel_size=(1, 1))
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(64, 6, 3)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        self.readout =nn.Linear(912, 1)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.layer1(x))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.layer2(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        # pdb.set_trace()
        x = self.readout(x)

        return x


class Net_cc(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_cc, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 6, 3)
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(6, 10, 3)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        self.readout = nn.Linear(240, 1)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.max_pool2d(F.relu(self.layer1(x)), (2, 2))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.layer2(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        # pdb.set_trace()
        x = self.readout(x)

        return x

class Net_ff(nn.Module):

    def __init__(self):
        super(Net_ff, self).__init__()
        # 1 input image channel, 6 output channels, 2 square convolution
        # kernel
        self.layer1 = nn.Linear(40*18,64)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.layer2 = nn.Linear(64, 64)  # from image dimension
        self.readout = nn.Linear(64, 1)

    def forward(self, x):
        # pdb.set_trace()
        x = torch.flatten(x, 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.readout(x)
        return x

class Net_fc5(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_fc5, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=64,
                                      kernel_size=(1, 1))
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(64, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        self.readout = nn.Linear(756, 1)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.layer1(x))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.layer2(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        # pdb.set_trace()
        x = self.readout(x)

        return x


class Net_fc7(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_fc7, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=64,
                                      kernel_size=(1, 1))
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(64, 6, 7)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        self.readout = nn.Linear(612, 1)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.layer1(x))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.layer2(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        # pdb.set_trace()
        x = self.readout(x)

        return x


class Net_cc5(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_cc5, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 6, 3)
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(6, 10, 5)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        self.readout = nn.Linear(140, 1)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.max_pool2d(F.relu(self.layer1(x)), (2, 2))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.layer2(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        # pdb.set_trace()
        x = self.readout(x)

        return x

class Net_cc7(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_cc7, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 6, 3)
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(6, 10, 7)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        self.readout = nn.Linear(60, 1)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.max_pool2d(F.relu(self.layer1(x)), (2, 2))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.layer2(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        # pdb.set_trace()
        x = self.readout(x)

        return x

class Net_smCC(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_smCC, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 6, 3)

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(6, 10, 3)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        self.readout = nn.Conv2d(10, 1, 3)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.max_pool2d(F.relu(self.layer1(x)), (2, 2))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.layer2(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1)
        x = self.readout(x)
        # pdb.set_trace()
        x = torch.flatten(x, 1).unsqueeze(1)
        x = F.max_pool1d(x, 6)
        x = torch.flatten(x, 1)
        return x


class Net_sCC(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_sCC, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 6, 3)

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(6, 10, 3)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        self.readout = nn.Conv2d(10, 1, 3)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.layer1(x))
        # Max pooling over a (2, 2) window
        x = F.relu(self.layer2(x))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1)
        x = self.readout(x)
        # pdb.set_trace()
        x = torch.flatten(x, 1).unsqueeze(1)
        x = F.max_pool1d(x, 408)
        x = torch.flatten(x, 1)
        return x


class Net_sCC5(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_sCC5, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 6, 5)

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(6, 10, 5)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        self.readout = nn.Conv2d(10, 1, 5)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.layer1(x))
        # Max pooling over a (2, 2) window
        x = F.relu(self.layer2(x))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1)
        x = self.readout(x)
        # pdb.set_trace()
        x = torch.flatten(x, 1).unsqueeze(1)
        x = F.max_pool1d(x, 168)
        x = torch.flatten(x, 1)
        return x


class Net_sCC7(nn.Module):

    def __init__(self):
        # an affine operation: y = Wx + b
        super(Net_sCC7, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 6, 2)

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.layer2 = nn.Conv2d(6, 10, 7)
        # self.conv2 = nn.Conv2d(16, 32, 3)
        self.readout = nn.Conv2d(10, 1, 7)

    def forward(self, x):
        # pdb.set_trace()
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.layer1(x))
        # Max pooling over a (2, 2) window
        x = F.relu(self.layer2(x))
        # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1)
        x = self.readout(x)
        # pdb.set_trace()
        x = torch.flatten(x, 1).unsqueeze(1)
        x = F.max_pool1d(x, 135)
        x = torch.flatten(x, 1)
        return x

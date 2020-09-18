from dependencies import *


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class CNNone(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.convolutional = nn.Conv2d(in_channels = input_channels, 
                                    kernel_size = 3, out_channels = output_channels,
                                     stride=1, padding=1)
        #self.batchn = nn.BatchNorm2d(num_features = output_channels) 
        self.swish = Swish()
        self.pool = nn.MaxPool2d(kernel_size = 2)
    def forward(self, data):
        output = self.convolutional(data)
        #output = self.batchn(output)
        output = self.swish(output)
        output = self.pool(output)
        return output



"""
x = torch.Tensor(np.random.random(size=(5, 523, 496)))
cnn = CNNcheck()
y = cnn(x)
print(y.size())
"""

import torch.nn as nn

"""# Архитектура сети

Класс Block - объединение двух слоев, основанное на логике сети ResNet

Класс ClassificatorNet - наша модель, состоящая из входного слоя, 5 блоков и выходного слоя.
"""

class Block(nn.Module):
  def __init__(self, num_chanals):
    super().__init__()

    self.convalution0 = nn.Conv2d(num_chanals, num_chanals, 3, padding = 1)
    self.batch_norm0 = nn.BatchNorm2d(num_chanals)
    self.activation = nn.LeakyReLU(0.2, inplace= True)
    self.convalution1 = nn.Conv2d(num_chanals, num_chanals, 3, padding = 1)
    self.batch_norm1 = nn.BatchNorm2d(num_chanals)

  def forward(self, x):
    result = self.convalution0(x)
    result = self.batch_norm0(result)
    result = self.activation(result)
    result = self.convalution1(result)
    result = self.batch_norm1(result)

    return self.activation(x + result)

class ClassificatorNet(nn.Module):
  def __init__(self,in_ch, num_ch, out_ch):
    super().__init__()
    self.conv0 =  nn.Conv2d(in_ch, num_ch, 3, stride=2, padding= 1)
    self.activation0 = nn.LeakyReLU(0.2, inplace= True)


    self.layer1 = Block(num_ch)
    self.conv1 = nn.Conv2d(num_ch, num_ch, 1, stride=1, padding=1)
    self.layer2 = Block(num_ch)
    self.conv2 = nn.Conv2d(num_ch, 2*num_ch, 3, stride=2, padding= 1)
    self.layer3 = Block(num_ch*2)
    self.conv3 = nn.Conv2d(2*num_ch, 4*num_ch, 3, stride =2 , padding =1)
    self.layer4 = Block(num_ch*4)
    self.conv4 = nn.Conv2d(4*num_ch, 8*num_ch, 3, stride =2 , padding =1)
    self.layer5 = Block(num_ch*8)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.flatten = nn.Flatten()
    self.linear = nn.Linear(8*num_ch, out_ch)
    self.soft = nn.Softmax(1)

  def forward(self, x):
    result = self.conv0(x)
    result = self.activation0(result)

    result = self.layer1(result)
    result = self.conv1(result)
    result = self.layer2(result)
    result = self.conv2(result)
    result = self.layer3(result)
    result = self.conv3(result)
    result = self.layer4(result)
    result = self.conv4(result)
    result = self.layer5(result)

    result = self.avgpool(result)
    result = self.flatten(result)
    result = self.linear(result)

    return self.soft(result)


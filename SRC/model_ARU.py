import torch
from torch import nn
import sys

#################################
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def layers_list(self):
        return [self.conv]

    def forward(self, x):
        x = self.conv(x)
        return x

#################################
class mid(nn.Module):
    def __init__(self, in_ch, out_ch, small_ch=None):
        super(mid, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        if small_ch is None:
            self.conv2 = nn.Conv2d(out_ch, in_ch, 3, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_ch, small_ch, 3, padding=1)

    def layers_list(self):
        return [self.conv1,self.conv2]

    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


#################################
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(double_conv, self).__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def layers_list(self):
        return [self.conv1,self.bn1,self.conv2,self.bn2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #return torch.sigmoid(x)
        #return torch.tanh(x)
        return x

#################################
class down(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super(down, self).__init__()
        self.pool = pool
        if pool:
            self.mp = nn.MaxPool2d(2)
        self.conv = double_conv(in_ch, out_ch)

    def layers_list(self):
        return self.conv.layers_list()

    def forward(self, x):
        if self.pool:
            x = self.mp(x)
        x = self.conv(x)
        return x

#################################
class up(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch):
        super(up, self).__init__()
#        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = double_conv(in_ch, out_ch, mid_ch=mid_ch)

    def layers_list(self):
        return self.conv.layers_list()

    def forward(self, x1, x2=None, x3=None):
        x1 = nn.functional.interpolate(x1,scale_factor=2, mode='nearest')
#        x1 = self.up(x1)
        if x2 is None and x3 is None:
            x = x1
        elif x3 is None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = torch.cat([x2, x3, x1], dim=1)
        x = self.conv(x)

        return x

class blstm(nn.Module):
    def __init__(self, num_channels, out_ch):
        super(blstm, self).__init__()
        self.lstmInpSz = int(num_channels * int(out_ch))  # divided by 2 because pooling
        self.lstm = nn.LSTM(input_size=self.lstmInpSz,num_layers=3, hidden_size=150, bidirectional=True, dropout=0.3)
        
    def forward(self,x):
        x = x.permute(3, 0, 1, 2)            # Tensor:WxBxCxH'        
        x = x.view(x.size(0), x.size(1), -1) # Tensor:WxBxF  F=(C*H')
        x,_=self.lstm(x)
        return x

class fc(nn.Module):
    def __init__(self, num_channels, out_ch):
        super(fc, self).__init__()
        self.fc = nn.Linear(in_features=150*2, out_features=out_ch)
        
    def forward(self,x):
        return self.fc(x)
    
#################################
class UNet(nn.Module):
    def __init__(self, input_ch, n_classes):
        super(UNet, self).__init__()
        self.down1 = down(input_ch, 64, pool=False)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.mid  = mid(512, 1024)
        self.up1 = up(1024, 256, 512)
        self.up2 = up(512, 128, 256)
        self.up3 = up(256, 64, 128)
        self.up4 = up(128, 64, 64)
        self.outc = outconv(64, n_classes)
        self.lstm = blstm(n_classes, n_classes)
        
        self.fc = fc(n_classes*2,n_classes)
        
    def forward(self, x):
        x1 = self.down1(x)        
        x2 = self.down2(x1)        
        x3 = self.down3(x2)        
        x4 = self.down4(x3)        
        x5 = self.mid(x4)
                
        x = self.up1(x5, x4)
        x = self.up2(x, x3)                
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.outc(x)        

        x = self.lstm(x)

        x = self.fc(x)

        x=torch.sigmoid(x)
        return x

#################################

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()        
        self.fc1 = nn.Linear(20 * 32 * 64, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):    
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


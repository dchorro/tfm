import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image,make_grid
import sys,os
from torch.nn.functional import relu
import torchvision.models as models

# References:
# B: mini-batch size
# C: number of channes (or feature maps)
# H: vertical dimension of feature map (or input image)
# W: horizontal dimension of feature map (or input image)
# K: number of different classes (unique chars)
class HTRModel_orig(nn.Module):
    def __init__(self, num_classes, line_height=64):
        super(HTRModel_orig, self).__init__()
        
        self.layer1 = nn.Sequential(        
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,2))
        )
      #  self.layer2 = nn.Sequential(
      #      nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1,1), padding=0),
      #      nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
      #      nn.BatchNorm2d(64),
      #      nn.ReLU(),
 #    #       nn.AvgPool2d(kernel_size=(2,1))
      #      nn.MaxPool2d(kernel_size=(2,1))
      #      #nn.LeakyReLU(),
      #  )
      #  self.layer3 = nn.Sequential(
      #      nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1,1), padding=0),
      #      nn.Conv2d(in_channels=2, out_channels=48, kernel_size=3, padding=1),
      #      nn.BatchNorm2d(48),
      #      nn.ReLU(),
#     #       nn.AvgPool2d(kernel_size=(2,1))
      #      #nn.LeakyReLU()
      #  )
      #  self.layer4 = nn.Sequential(        
      #      nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1),
      #      nn.ReLU(),            
      #      nn.AvgPool2d(kernel_size=(2,1))
      #      #nn.LeakyReLU(),
      #  )
        
        num_of_channels=48
        lstmInpSz = 12800  #int(num_of_channels * line_height//2)#//2//2//2)  # divided by 2 because pooling
        self.lstm = nn.LSTM(input_size=lstmInpSz,num_layers=3, hidden_size=100, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(in_features=100*2, out_features=num_classes)
#        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)


    def forward(self, x):      # x ---> Tensor:NxCxHxW,  C=1 for binary/grey-scale images
        x = self.layer1(x)  # Tensor:NxCxH_1xW, H_1=H/2 because of maxpooling)
#        x = self.layer2(x)  # Tensor:NxCxH_2xW, H_2=H/2/2
#        x = self.layer3(x)  # Tensor:NxCxH_2xW, H_2=H/2/2
#        x = self.layer4(x)  # Tensor:NxCxH_3xW, H_3=H/2/2/2
        
        x = x.permute(3, 0, 1, 2)            # Tensor:WxNxCxH'        
        x = x.view(x.size(0), x.size(1), -1) # Tensor:WxNxF  F=(C*H') 
        x, _ = self.lstm(x)                  # Tensor:WxNx(2*250),
                                             # 2 as bidireccional=True was set
                                             # 250 is the number of LSTM units  
        x = self.fc(x)        # Tensor:WxNxK, K=number of different char classes
 #       x = self.fc2(x)
        x=torch.sigmoid(x)  # for BCELoss()
        return x



class HTRModel_0(nn.Module):
    def __init__(self, line_height=64):
        super(HTRModel_0, self).__init__()
        
        self.line_height = line_height
        self.cont=0
        
        self.layer1 = nn.Sequential(        
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
           
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.layer2 = nn.Sequential(                        
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer3 = nn.Sequential(        
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
          
            nn.Conv2d(in_channels=16, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
           
        )

        self.num_of_channels=512
        self.lstmInpSz = int(self.num_of_channels * int(line_height//2//2))  # divided by 2 because pooling

        self.dropout = nn.Dropout(0.25)
        
        self.lstm = nn.LSTM(input_size=self.lstmInpSz,num_layers=3, hidden_size=150, bidirectional=True)
        self.fc = nn.Linear(in_features=150*2, out_features=line_height)
       
    def forward(self, x):      # x ---> Tensor:BxCxHxW,
        x = self.layer1(x)  # Tensor:BxCxH_1xW, H_1=H/2 because of maxpooling)
        x = self.layer2(x)  # Tensor:BxCxH_2xW, H_2=H/2/2
        x = self.layer3(x)  # Tensor:BxCxH_2xW, H_2=H/2/2
        x = x.permute(3, 0, 1, 2)            # Tensor:WxBxCxH'        
        x = x.view(x.size(0), x.size(1), -1) # Tensor:WxBxF  F=(C*H')
        x, _ = self.lstm(x)                  # Tensor:WxBx(2*250),
                                             # 2 as bidireccional=True was set
                                             # 250 is the number of LSTM units

        x = self.dropout(x)
        x = self.fc(x)        # Tensor:WxBxK, K=number of different char classes
        # print("-------------------------")
        # print(x.shape)
 #       for i in range(x.shape[0]):
 #           print(x[i][0][10].item())
        x=torch.sigmoid(x)  # for BCELoss()
        # print(x.shape)
        # print("-------------------------")
 #       x = torch.softmax(x)
        return x


    def get_feat_maps(self, x, out_dir, file_names):
        x = self.layer1(x)  # Tensor:BxCxH_1xW, H_1=H/2 because of maxpooling)
#        x = self.layer2(x)  # Tensor:BxCxH_2xW, H_2=H/2/2
     #   x = self.layer3(x)  # Tensor:BxCxH_2xW, H_2=H/2/2
     #   x = self.layer4(x)  # Tensor:BxCxH_3xW, H_3=H/2/2/2

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)        
       
        for b in range(int (x.size()[0])):
            pic=[]
            for i in range(int (x.size()[1])):  ## tot el batch
                if i < 10:
                    pic.append(1-x[b][i])

            pic = torch.stack(pic)
            pic = pic[:,None,:,:]

            pic = make_grid(pic,nrow=2,normalize=True,scale_each=True)
            file=out_dir+"/"+os.path.basename(file_names[b])+"_features_"+str(self.cont)+".png"
            save_image(pic, file, nrow=1);
        self.cont = self.cont+1


class HTRModel_ResNet(nn.Module):
    def __init__(self, line_height=64):
        super(HTRModel_ResNet, self).__init__()
        
        self.line_height = line_height
        self.cont = 0

        # Load pretrained ResNet and modify the first conv layer
        resnet = models.resnet18(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last FC and pooling layers
        # self.resnet_layers = self.resnet_layers
        print("--------------------------")
        # print(self.resnet_layers)
        print(self.resnet_layers[:6])
        print("--------------------------")
        self.map = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        
        # Modify the first conv layer to accept 1 channel input instead of 3
        # self.resnet_layers[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Calculate the feature size after the ResNet layers
        self.num_of_channels = 512  # Number of output channels of the last resnet layer

        # Create a dummy input to calculate the size after the ResNet layers
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, 1, line_height, 1000)  # 1000 is an arbitrary width
        #     cnn_output_size = self.resnet_layers(dummy_input).shape
        
        self.lstmInpSz = 300

        self.dropout = nn.Dropout(0.25)
        
        self.lstm = nn.LSTM(input_size=self.lstmInpSz, num_layers=3, hidden_size=150, bidirectional=True)
        self.fc = nn.Linear(in_features=150 * 2, out_features=line_height)

    def forward(self, x):  # x ---> Tensor:BxCxHxW,
        batch_size = x.size(0)
        x = self.map(x)
        x = self.resnet_layers(x)  # Tensor:BxCxH_1xW, H_1 is calculated by resnet layers
        x = x.permute(3, 0, 1, 2)  # Tensor:WxBxCxH'
        x = x.view(x.size(0), x.size(1), -1)  # Tensor:WxBxF  F=(C*H')
        x, _ = self.lstm(x)  # Tensor:WxBx(2*150), 2 as bidirectional=True was set, 150 is the number of LSTM units

        x = self.dropout(x)
        x = self.fc(x)  # Tensor:WxBx(line_height)
        # x = x.permute(1, 2, 0).contiguous()  # Tensor:Bx(line_height)xW
        return x

    


            

class UNet(nn.Module):
    def __init__(self, n_class=2):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = nn.ReLU()(self.e11(x))
        xe12 = nn.ReLU()(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = nn.ReLU()(self.e21(xp1))
        xe22 = nn.ReLU()(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = nn.ReLU()(self.e31(xp2))
        xe32 = nn.ReLU()(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = nn.ReLU()(self.e41(xp3))
        xe42 = nn.ReLU()(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = nn.ReLU()(self.e51(xp4))
        xe52 = nn.ReLU()(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu1 = nn.functional.pad(xu1, [0, xe42.size(3) - xu1.size(3), 0, xe42.size(2) - xu1.size(2)])
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = nn.ReLU()(self.d11(xu11))
        xd12 = nn.ReLU()(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu2 = nn.functional.pad(xu2, [0, xe32.size(3) - xu2.size(3), 0, xe32.size(2) - xu2.size(2)])
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = nn.ReLU()(self.d21(xu22))
        xd22 = nn.ReLU()(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu3 = nn.functional.pad(xu3, [0, xe22.size(3) - xu3.size(3), 0, xe22.size(2) - xu3.size(2)])
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = nn.ReLU()(self.d31(xu33))
        xd32 = nn.ReLU()(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu4 = nn.functional.pad(xu4, [0, xe12.size(3) - xu4.size(3), 0, xe12.size(2) - xu4.size(2)])
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = nn.ReLU()(self.d41(xu44))
        xd42 = nn.ReLU()(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        # out = F.softmax(out, 1)

        return out

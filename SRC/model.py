import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image,make_grid
import sys,os

# References:
# B: mini-batch size
# C: number of channes (or feature maps)
# H: vertical dimension of feature map (or input image)
# W: horizontal dimension of feature map (or input image)
# K: number of different classes (unique chars)

class HTRModel(nn.Module):
    def __init__(self, line_height=64):
        super(HTRModel, self).__init__()
        
        self.line_height = line_height
        self.cont=0
        
        self.conv = nn.Sequential(        
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,1)),
            
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
           
            nn.MaxPool2d(kernel_size=(2,1)),

            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
           
        self.num_of_channels=256
        self.lstmInpSz = int(self.num_of_channels * int(line_height)//2//2)  # divided by 2 because pooling    
#        self.lstmInpSz = 67584
        self.lstm = nn.LSTM(input_size=self.lstmInpSz,num_layers=3, hidden_size=150, bidirectional=True, dropout=0.5)
        
        self.fc1 = nn.Linear(in_features=150*2, out_features=line_height)#line_height)
#        self.fc2 = nn.Linear(in_features=1000, out_features=line_height)

        
    def forward(self, x):      # x ---> Tensor:BxCxHxW,
        x = self.conv(x)

        x = x.permute(3, 0, 1, 2)            # Tensor:WxBxCxH'        
        x = x.view(x.size(0), x.size(1), -1) # Tensor:WxBxF  F=(C*H')
        x, _  = self.lstm(x)

        x = self.fc1(x)
#        x = self.fc2(x)
        
        x=torch.sigmoid(x)  # for BCELoss()
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


            
class HTRModel2D(nn.Module):
    def __init__(self, line_height=64):
        super(HTRModel2D, self).__init__()
        
        self.line_height = line_height
        self.cont=0
        
        self.layer01 = nn.Sequential(        
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
           
            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer02 = nn.Sequential(        
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer03 = nn.Sequential(        
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
           
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            #nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer04 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
           
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer05 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
           
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
          
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
           
        )
        self.layer11 = nn.Sequential(        
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
           
            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer12 = nn.Sequential(        
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer13 = nn.Sequential(        
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
           
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            #nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
           
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2,1))
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
           
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
          
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
           
        )
        self.num_of_channels=512
#        self.lstmInpSz = int(self.num_of_channels * int(line_height)//2//2//2//2)  # divided by 2 because pooling    
        self.lstmInpSz =  51200
        self.dropout = nn.Dropout(0.25)
        
        self.lstm1 = nn.LSTM(input_size=self.lstmInpSz,num_layers=1, hidden_size=50, bidirectional=True, dropout=0)
        self.lstm2 = nn.LSTM(input_size=self.lstmInpSz,num_layers=1, hidden_size=50, bidirectional=True, dropout=0)
        
        #self.fc1 = nn.Linear(in_features=100*2, out_features=line_height)
        #self.fc2 = nn.Linear(in_features=100*2, out_features=line_height)
        #self.fc3 = nn.Linear(in_features=line_height*2, out_features=line_height)
        self.fc3 = nn.Linear(in_features=(50*2*2), out_features=1000)
        self.fc4 = nn.Linear(in_features=1000, out_features=line_height)
    def forward(self, x):      # x ---> Tensor:BxCxHxW,
        x2 = torch.rot90(x,3,[2,3])
                        
        x = self.layer01(x)  # Tensor:BxCxH_1xW, H_1=H/2 because of maxpooling)        
        x = self.layer02(x)  # Tensor:BxCxH_2xW, H_2=H/2/2
        x = self.layer03(x)  # Tensor:BxCxH_2xW, H_2=H/2/2
#        x = self.layer04(x)  # Tensor:BxCxH_3xW, H_3=H/2/2/2
#        x = self.layer05(x)
     
        x2 = self.layer11(x2)  # Tensor:BxCxH_1xW, H_1=H/2 because of maxpooling)         
        x2 = self.layer12(x2)  # Tensor:BxCxH_2xW, H_2=H/2/2
        x2 = self.layer13(x2)  # Tensor:BxCxH_2xW, H_2=H/2/2
#        x2 = self.layer4(x2)  # Tensor:BxCxH_3xW, H_3=H/2/2/2
#        x2 = self.layer5(x2)

        x = x.permute(3, 0, 1, 2)            # Tensor:WxBxCxH'        
#        x = x.view(x.size(0), x.size(1), -1) # Tensor:WxBxF  F=(C*H')

        x2 = x2.permute(3,0,1,2)        
#        x2 = x.view(x2.size(0), x2.size(1), -1)

        x  =  x.flatten(start_dim=2)
        x2 = x2.flatten(start_dim=2)
#        torch.reshape(x2,(x2.size(0),x2.size(1),-1))
        
        x, _  = self.lstm1(x)
        x2, _ = self.lstm2(x2)                 # Tensor:WxBx(2*100),
                                             # 2 as bidireccional=True was set
      
        x = self.dropout(x)
       # x = self.fc1(x)        # Tensor:WxBxK, K=number of different char classes

        x2 = self.dropout(x2)
      #  x2 = self.fc2(x2)


        x = torch.cat((x,x2),dim=2)

        x = self.fc3(x)
        x = self.fc4(x)
        
        x=torch.sigmoid(x)  # for BCELoss()
        return x


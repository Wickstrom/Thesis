import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN32(nn.Module):
    def __init__(self,drop,activation):
        super(FCN32,self).__init__()
        
        self.drop = drop
        self.activation = activation

        self.features = nn.Sequential(
                
                # Conv 1
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(3,64,3,1,1),              
                self.activation,
                nn.BatchNorm2d(64),  
                
                # Conv 2 + pooling 
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(64,64,3,1,1),
                nn.MaxPool2d(2),
                self.activation,
                nn.BatchNorm2d(64),        
                
                # Conv 3
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(64,128,3,1,1),               
                self.activation,
                nn.BatchNorm2d(128), 

                # Conv 4 + pooling
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(128,128,3,1,1),
                nn.MaxPool2d(2),
                self.activation,
                nn.BatchNorm2d(128), 
                
                # Conv 5
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(128,256,3,1,1),              
                self.activation,
                nn.BatchNorm2d(256),
                
                # Conv 6
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(256,256,3,1,1),              
                self.activation,
                nn.BatchNorm2d(256),
                
                # Conv 7 + pooling
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(256,256,3,1,1),
                nn.MaxPool2d(2),
                self.activation,
                nn.BatchNorm2d(256),     
                
                # Conv 8
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(256,512,3,1,1),              
                self.activation,
                nn.BatchNorm2d(512),
                
                # Conv 9
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(512,512,3,1,1),              
                self.activation,
                nn.BatchNorm2d(512),
                
                # Conv 10 + pooling
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(512,512,3,1,1),
                nn.MaxPool2d(2),
                self.activation,
                nn.BatchNorm2d(512),    
                     
                # Conv 11
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(512,512,3,1,1),              
                self.activation,
                nn.BatchNorm2d(512),
                
                # Conv 12
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(512,512,3,1,1),              
                self.activation,
                nn.BatchNorm2d(512),
                
                # Conv 13 + pooling
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(512,512,3,1,1),
                nn.MaxPool2d(2),
                self.activation,
                nn.BatchNorm2d(512),  
                
                # Conv 14
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(512,4096,7,1,3),              
                self.activation,
                nn.BatchNorm2d(4096),
                
                # Conv 15
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(4096,4096,1,1,0),              
                self.activation,
                nn.BatchNorm2d(4096),
                                
                # Conv 16
                nn.Dropout2d(p=self.drop),
                nn.Conv2d(4096,2,1,1,0),           
                             
                # Upsampling ( Transposed convolution)
                nn.ConvTranspose2d(2,2,64,32,16)
                )
        
    def forward(self,x):
        
        out = self.features(x)
        
        out = out.permute(1,0,2,3).contiguous()
        out = out.view(2,-1)
        out = out.permute(1,0)
      
        return out
    
    def accuracy(self,x,y):
		
        y_pred = torch.max(F.softmax(self.forward(x)),1)[1]
        accuracy = torch.mean(torch.eq(y,y_pred).float())
        
        return(accuracy)
    
    def IoU(self,x,y,a):
        
        y_pred = torch.max(F.softmax(self.forward(x)),1)[1]
        TP = torch.sum(((y == a).float() * (y == y_pred).float()).float()) 
        FP = torch.sum(((y != a).float() * (y_pred == a).float()).float()) 
        FN = torch.sum(((y == a).float() * (y != y_pred).float()).float()) 
        
        return TP.float() / (TP + FP + FN).float()

    def F1(self,x,y,a):
        
        y_pred = torch.max(F.softmax(self.forward(x)),1)[1]
        TP = torch.sum(((y == a).float() * (y == y_pred).float()).float()) 
        FP = torch.sum(((y != a).float() * (y_pred == a).float()).float()) 
        FN = torch.sum(((y == a).float() * (y != y_pred).float()).float()) 
        
        return 2 * TP.float() / (2*TP + FP + FN).float()
    
    def MFB(self,y):
        
        f0 = torch.sum(torch.eq(y,0).float())
        f1 = torch.sum(torch.eq(y,1).float())
        mean_freq = (f0+f1) / 2
        w0 = mean_freq / f0
        w1 = mean_freq / f1
        weights = torch.cat([w0,w1])
        
        return weights
        
        
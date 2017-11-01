import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self,n_in,k,activation):
        super(Layer,self).__init__()  

        self.k = k
        self.activation = activation        
        self.bn = nn.BatchNorm2d(n_in)
        self.drop = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(n_in,k,kernel_size=3,padding=1)
   
    def forward(self,x):
        out = self.drop(self.conv(self.activation(self.bn(x))))
        return (torch.cat([out,x],1))      
    
class TransitionDown(nn.Module):
    def __init__(self,n_in,activation):
        super(TransitionDown,self).__init__()  
        
        self.n_in = n_in
        self.activation = activation
        
        self.block = nn.Sequential(
            *([nn.BatchNorm2d(self.n_in),
               self.activation,
               nn.Conv2d(self.n_in, self.n_in, kernel_size=1),
               nn.Dropout2d(p=0.2),
               nn.MaxPool2d(2,stride=2)])
        )
    def forward(self,x):
        return (self.block(x))
    

class FCDensenet(nn.Module):
    def __init__(self,num_classes,k,activation):
        super(FCDensenet,self).__init__()
        
        self.k = k
        self.activation = activation
        self.num_classes = num_classes
        
        self.block0 = nn.Sequential(*([nn.Conv2d(3,48,3,padding=1)]))
    
        self.n_feat1 = 4*self.k + 48 
        self.block1 = self.block(4,48,self.k)
        self.td1 = TransitionDown(self.n_feat1,self.activation)
        
        self.n_feat2 = 5*self.k + self.n_feat1
        self.block2 = self.block(5,self.n_feat1,self.k)
        self.td2 = TransitionDown(self.n_feat2,self.activation)
        
        self.n_feat3 = 7*self.k + self.n_feat2        
        self.block3 = self.block(7,self.n_feat2,self.k)
        self.td3 = TransitionDown(self.n_feat3,self.activation)

        self.n_feat4 = 10*self.k + self.n_feat3        
        self.block4 = self.block(10,self.n_feat3,self.k)
        self.td4 = TransitionDown(self.n_feat4,self.activation)
        
        self.n_feat5 = 12*self.k + self.n_feat4        
        self.block5 = self.block(12,self.n_feat4,self.k)
        self.td5 = TransitionDown(self.n_feat5,self.activation)
        
        self.n_feat6 = 15*self.k + self.n_feat5        
        self.block6 = self.block(15,self.n_feat5,self.k)
        
        self.n_feat7 = self.n_feat5 + 15*self.k
        self.tu5 = nn.ConvTranspose2d(15*self.k,15*self.k,
                                      kernel_size=4,stride=2,padding=1,bias=False)
        self.block7 = self.block(12,self.n_feat7,self.k)
        
        self.n_feat8 = self.n_feat4 + 12*self.k
        self.tu4 = nn.ConvTranspose2d(12*self.k,12*self.k,
                                      kernel_size=4,stride=2,padding=1,bias=False)
        self.block8 = self.block(10,self.n_feat8,self.k)
        
        self.n_feat9 = self.n_feat3 + 10*self.k
        self.tu3 = nn.ConvTranspose2d(10*self.k,10*self.k,
                                      kernel_size=4,stride=2,padding=1,bias=False)
        self.block9 = self.block(7,self.n_feat9,self.k)

        self.n_feat10 = self.n_feat2 + 7*self.k
        self.tu2 = nn.ConvTranspose2d(7*self.k,7*self.k,
                                      kernel_size=4,stride=2,padding=1,bias=False)
        self.block10 = self.block(5,self.n_feat10,self.k)
        
        self.n_feat11 = self.n_feat1 + 5*self.k
        self.tu1 = nn.ConvTranspose2d(5*self.k,5*self.k,
                                      kernel_size=4,stride=2,padding=1,bias=False)
        self.block11 = self.block(4,self.n_feat11,self.k)

        self.n_feat11 = self.n_feat11 + 4*self.k        
        self.block12 = nn.Sequential(*([nn.Conv2d(self.n_feat11,
                                                  self.num_classes,1,padding=0)]))

    def forward(self,x):
        
        block0 = self.block0(x)
        block1 = self.block1(block0)
        td1 = self.td1(block1)
        block2 = self.block2(td1)
        td2 = self.td2(block2)
        block3 = self.block3(td2)
        td3 = self.td3(block3)
        block4 = self.block4(td3)
        td4 = self.td4(block4)
        block5 = self.block5(td4)
        td5 = self.td5(block5)
        block6 = self.block6(td5)        
        tu5 = torch.cat([self.tu5(block6[:,self.n_feat5:,:,:]),block5],1)
        block7 = self.block7(tu5)
        tu4 = torch.cat([self.tu4(block7[:,self.n_feat7:,:,:]),block4],1)
        block8 = self.block8(tu4)
        tu3 = torch.cat([self.tu3(block8[:,self.n_feat8:,:,:]),block3],1)
        block9 = self.block9(tu3) 
        tu2 = torch.cat([self.tu2(block9[:,self.n_feat9:,:,:]),block2],1)
        block10 = self.block10(tu2) 
        tu1 = torch.cat([self.tu1(block10[:,self.n_feat10:,:,:]),block1],1)
        block11 = self.block11(tu1)
        block12 = self.block12(block11)
        
        out = block12.permute(1,0,2,3).contiguous()
        out = out.view(2,-1)
        out = out.permute(1,0)
        
        return out
    
    def block(self,l,n_in,k):
        layers = []
        for i in range(l):
            layers.append(Layer(n_in,k,self.activation))
            n_in += k            
        return nn.Sequential(*layers)
    
    
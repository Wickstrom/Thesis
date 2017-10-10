import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from polyp_loader import transform
from torch.autograd import Variable


class FCN8(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN8, self).__init__()
        
        features = list(models.vgg16_bn().features.children())

        self.features3 = nn.Sequential(*features[: 24])
        self.features4 = nn.Sequential(*features[24: 34])
        self.features5 = nn.Sequential(*features[34:])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.require_grad = False
            
            if isinstance(m,nn.BatchNorm2d):
                m.require_grad = False


        
        self.fc = nn.Sequential(
                nn.Conv2d(512,4096,kernel_size=7,padding=3),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(4096,4096,kernel_size=1),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                )
 
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)   
        self.score_pool5 = nn.Conv2d(4096,num_classes,kernel_size=1)

        self.upsample_pool5 = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=4,stride=2,padding=1,bias=False)
        self.upsample_pool4 = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=4,stride=2,padding=1,bias=False)
        self.upsample_pool3 = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=16,stride=8,padding=4,bias=False)


    def forward(self, x):
        
        pool_3 = self.features3(x)
        pool_4 = self.features4(pool_3)
        pool_5 = self.features5(pool_4)
        fc = self.fc(pool_5) 
        
        score_pool3 = self.score_pool3(pool_3)
        score_pool4 = self.score_pool4(pool_4)
        score_pool5 = self.score_pool5(fc)
        
        out = self.upsample_pool5(score_pool5)
        out = 2 * out + score_pool4
        out = self.upsample_pool4(out)
        out = 2 * out + score_pool3
        out = self.upsample_pool3(out)
        
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
            
    def tr(self,x,y,bs,n_train_batch,optimizer,criterion,model):
        
        index = np.random.permutation(len(x))  
        model.train()        
        c = []
        
        for index_train in range(n_train_batch): 
            
            x_tr,y_tr = transform(x[index[index_train]][0],y[index[index_train]][0])         
            x_tr = Variable(x_tr.float(),requires_grad=True) .cuda()                                                                                          
            y_tr = y_tr.view(-1,x_tr.size(2)*x_tr.size(3))
            y_tr = Variable(y_tr.long()).cuda()         
            optimizer.zero_grad()                
            output = self.forward(x_tr)
            loss_tr = criterion(output,y_tr[0])
            loss_tr.backward()
            optimizer.step()
            c.append((loss_tr.data[0]))           
        return c
    
    def va(self,x,y,bs,n_valid_batch,model):
        
        index = np.random.permutation(len(x)) 
        model.eval()
        v = []
        for index_test in range(n_valid_batch): 
            
            x_va = x[index[index_test]]
            y_va = y[index[index_test]]    
            x_va = Variable(x_va.float()).cuda()     
                                                                      
            y_va = y_va.view(-1,x_va.size(2)*x_va.size(3))
            y_va = Variable(y_va.long()).cuda()
            pp = self.accuracy(x_va,y_va[0])
            IoU_1 = self.IoU(x_va,y_va[0],1)
            IoU_0 = self.IoU(x_va,y_va[0],0)
            IoU_m = (IoU_1.data[0]+IoU_0.data[0]) / 2
            v.append((pp.data[0],IoU_1.data[0],IoU_0.data[0],IoU_m))
        return v
    
    def te(self,x,y,bs,n_test_batch,model):
        
        index = np.random.permutation(len(x)) 
        model.eval()
        l = []
        for index_test in range(n_test_batch): 
            
            x_te = x[index[index_test]]
            y_te = y[index[index_test]]       
                       
            x_te = Variable(x_te.float()).cuda()     
                                                                      
            y_te = y_te.view(-1,x_te.size(2)*x_te.size(3))
            y_te = Variable(y_te.long()).cuda()
            pp = self.accuracy(x_te,y_te[0])
            IoU_1 = self.IoU(x_te,y_te[0],1)
            IoU_0 = self.IoU(x_te,y_te[0],0)
            IoU_m = (IoU_1.data[0]+IoU_0.data[0]) / 2
            l.append((pp.data[0],IoU_1.data[0],IoU_0.data[0],IoU_m))
        return l
    

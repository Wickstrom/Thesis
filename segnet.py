import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from polyp_loader import transform
from torch.autograd import Variable

class SegNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SegNet, self).__init__()
        features = list(models.vgg16_bn().features.children())
        
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.return_indices=True

        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:24])
        self.enc4 = nn.Sequential(*features[24:34])
        self.enc5 = nn.Sequential(*features[40:])
        
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.require_grad = False            
            if isinstance(m,nn.BatchNorm2d):
                m.require_grad = False        

        self.dec5 = nn.Sequential(
            *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 3)
        )
        self.dec4 = nn.Sequential(
            *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 2)
        )
        self.dec4_out = nn.Sequential(
            *([nn.Conv2d(512, 256, kernel_size=3, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True)])
        )
        self.dec3 = nn.Sequential(
            *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True)] * 2)
        )
        self.dec3_out = nn.Sequential(
            *([nn.Conv2d(256, 128, kernel_size=3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(inplace=True)])
        )
        self.dec2 = nn.Sequential(
            *([nn.Conv2d(128, 128, kernel_size=3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(inplace=True)] * 2)
        )
        self.dec2_out = nn.Sequential(
            *([nn.Conv2d(128, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True)])
        )
        self.dec1 = nn.Sequential(
            *([nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True)] * 2)
        )
        self.dec1_out = nn.Sequential(
            *([nn.Conv2d(64, 2, kernel_size=3, padding=1)])
        )

    def forward(self, x):
        enc1, idx1 = self.enc1(x)
        enc2, idx2 = self.enc2(enc1)
        enc3, idx3 = self.enc3(enc2)
        enc4, idx4 = self.enc4(enc3)
        enc5, idx5 = self.enc5(enc4)

        dec5 = self.dec5(F.max_unpool2d(enc5,idx5,kernel_size=2,stride=2))
        dec4 = self.dec4(F.max_unpool2d(dec5,idx4,kernel_size=2,stride=2))
        dec4_out = self.dec4_out(dec4)
        dec3 = self.dec3(F.max_unpool2d(dec4_out,idx3,kernel_size=2,stride=2))
        dec3_out = self.dec3_out(dec3)
        dec2 = self.dec2(F.max_unpool2d(dec3_out,idx2,kernel_size=2,stride=2))
        dec2_out = self.dec2_out(dec2)
        dec1 = self.dec1(F.max_unpool2d(dec2_out,idx1,kernel_size=2,stride=2))
        dec1_out = self.dec1_out(dec1)
                
        out = dec1_out.permute(1,0,2,3).contiguous()
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
        
        for index_train in range (n_train_batch):
                
            rand_init = np.random.randint(0,len(x),1)[0]
            batch_x, batch_y = transform(x[rand_init][0], y[rand_init][0])                       
            for ix in index[index_train*bs:bs*(index_train)+bs]:             
                x_temp, y_temp  = transform(x[ix][0],y[ix][0])
                batch_x = torch.cat([batch_x,x_temp])
                batch_y = torch.cat([batch_y,y_temp])
                       
            x_tr = Variable(batch_x.float(),requires_grad=True).cuda()         
            y_tr = batch_y.view(-1,x_tr.size(2)*x_tr.size(3))
            y_tr = Variable(y_tr.long()).cuda()      
            optimizer.zero_grad()                
            output = self.forward(x_tr)
            loss_tr = criterion(output,y_tr.view(-1))
            loss_tr.backward()
            optimizer.step()
            c.append((loss_tr.data[0]))           
        return c
        
    def va(self,x,y,n_valid_batch,model):
        
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
    
    def te(self,x,y,n_test_batch,model):
        
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



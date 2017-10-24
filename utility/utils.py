import torch
import numyp as np
import torch.nn.functional as F
from polyp_loader import transform
from torch.autograd import Variable

#####################################################################################################
# This file contains a collection of small function for my networks. Note that all functions        #
# assumes that the y input(label) is of shape (Number of Samples , umber of Classes).              #
#####################################################################################################

    
def accuracy(x,y,model):                                              # Compute mean per-pixel accuracy.
    
    y_pred = torch.max(F.softmax(model(x)),1)[1]
    accuracy = torch.mean(torch.eq(y,y_pred).float())
        
    return(accuracy)
    
    
def IoU(x,y,a,model):
        
    y_pred = torch.max(F.softmax(model(x)),1)[1]                      # Compute IoU-score for class a.
    TP = torch.sum(((y == a).float() * (y == y_pred).float()).float()) 
    FP = torch.sum(((y != a).float() * (y_pred == a).float()).float()) 
    FN = torch.sum(((y == a).float() * (y != y_pred).float()).float()) 
        
    return TP.float() / (TP + FP + FN).float()

def F1(x,y,a,model):
        
    y_pred = torch.max(F.softmax(model(x)),1)[1]                      # Compute F1-score for class a.
    TP = torch.sum(((y == a).float() * (y == y_pred).float()).float()) 
    FP = torch.sum(((y != a).float() * (y_pred == a).float()).float()) 
    FN = torch.sum(((y == a).float() * (y != y_pred).float()).float()) 
        
    return 2 * TP.float() / (2*TP + FP + FN).float()

def MFB(y):                                                           # Median frequency balancing (https://arxiv.org/abs/1411.4734)
                                                                      # for two classes.                                                                        
    f0 = torch.sum(torch.eq(y,0).float())
    f1 = torch.sum(torch.eq(y,1).float())
    mean_freq = (f0+f1) / 2
    w0 = mean_freq / f0
    w1 = mean_freq / f1
    weights = torch.cat([w0,w1])
        
    return weights

####################################################################################
# Training function for all networks. Detailed explentation coming soon.           #
            
def train(x,y,bs,n_train_batch,optimizer,criterion,model,
           crop,rot,shear,zoom,t):
        
    index = np.random.permutation(len(x))
    model.train()        
    c = []
        
    for index_train in range (n_train_batch):
                
        rand_init = np.random.randint(0,len(x),1)[0]
        batch_x, batch_y = transform(x[rand_init][0], y[rand_init][0],
                                         crop,rot,shear,zoom,t)                       
        for ix in index[index_train*bs:bs*(index_train)+bs]:             
            x_temp, y_temp  = transform(x[ix][0],y[ix][0],
                                         crop,rot,shear,zoom,t)
            batch_x = torch.cat([batch_x,x_temp])
            batch_y = torch.cat([batch_y,y_temp])
                       
        x_tr = Variable(batch_x.float(),requires_grad=True).cuda()         
        y_tr = batch_y.view(-1,x_tr.size(2)*x_tr.size(3))
        y_tr = Variable(y_tr.long()).cuda()      
        optimizer.zero_grad()                
        output = model(x_tr)
        loss_tr = criterion(output,y_tr.view(-1))
        loss_tr.backward()
        optimizer.step()
        c.append((loss_tr.data[0]))           
    return c
        
def valid(x,y,n_valid_batch,model):
        
    index = np.random.permutation(len(x)) 
    model.eval()
    v = []
    for index_test in range(n_valid_batch): 
            
        x_va = x[index[index_test]]
        y_va = y[index[index_test]]    
        x_va = Variable(x_va.float()).cuda()     
                                                                      
        y_va = y_va.view(-1,x_va.size(2)*x_va.size(3))
        y_va = Variable(y_va.long()).cuda()
        pp = accuracy(x_va,y_va[0])
        IoU_1 = IoU(x_va,y_va[0],1)
        IoU_0 = IoU(x_va,y_va[0],0)
        IoU_m = (IoU_1.data[0]+IoU_0.data[0]) / 2
        v.append((pp.data[0],IoU_1.data[0],IoU_0.data[0],IoU_m))
    return v
    
def test(x,y,n_test_batch,model):
        
    index = np.random.permutation(len(x)) 
    model.eval()
    l = []
    for index_test in range(n_test_batch): 
            
        x_te = x[index[index_test]]
        y_te = y[index[index_test]]       
                       
        x_te = Variable(x_te.float()).cuda()     
                                                                      
        y_te = y_te.view(-1,x_te.size(2)*x_te.size(3))
        y_te = Variable(y_te.long()).cuda()
        pp = accuracy(x_te,y_te[0])
        IoU_1 = IoU(x_te,y_te[0],1)
        IoU_0 = IoU(x_te,y_te[0],0)
        IoU_m = (IoU_1.data[0]+IoU_0.data[0]) / 2
        l.append((pp.data[0],IoU_1.data[0],IoU_0.data[0],IoU_m))
    return l

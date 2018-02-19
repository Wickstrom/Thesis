import torch
import numpy as np
import torch.nn.functional as F
from polyp_loader import transform
from torch.autograd import Variable


def accuracy(x, y, model):

    y_pred = torch.max(F.softmax(model(x)), 1)[1]
    accuracy = torch.mean(torch.eq(y, y_pred).float())

    return(accuracy)


def IoU(x,y,a,model):
        
    y_pred = torch.max(F.softmax(model(x)),1)[1]
    TP = torch.sum(((y == a).float() * (y == y_pred).float()).float()) 
    FP = torch.sum(((y != a).float() * (y_pred == a).float()).float()) 
    FN = torch.sum(((y == a).float() * (y != y_pred).float()).float()) 
        
    return TP.float() / (TP + FP + FN).float()

def F1(x,y,a,model):
        
    y_pred = torch.max(F.softmax(model(x)),1)[1]
    TP = torch.sum(((y == a).float() * (y == y_pred).float()).float()) 
    FP = torch.sum(((y != a).float() * (y_pred == a).float()).float()) 
    FN = torch.sum(((y == a).float() * (y != y_pred).float()).float()) 
        
    return 2 * TP.float() / (2*TP + FP + FN).float()

def MFB(y):
        
    f0 = torch.sum(torch.eq(y,0).float())
    f1 = torch.sum(torch.eq(y,1).float())
    mean_freq = (f0+f1) / 2
    w0 = mean_freq / f0
    w1 = mean_freq / f1
    weights = torch.cat([w0,w1])
        
    return weights
            
def train(x,y,bs,n_train_batch,optimizer,criterion,model,
           crop,rot,shear,zoom,t,cuda):
        
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
        
        if cuda == True:
            x_tr = Variable(batch_x.float(),requires_grad=True).cuda()
        else:
            x_tr = Variable(batch_x.float(),requires_grad=True)            
        y_tr = batch_y.view(-1,x_tr.size(2)*x_tr.size(3))
        if cuda == True:
            y_tr = Variable(y_tr.long()).cuda() 
        else:
            y_tr = Variable(y_tr.long())              
        optimizer.zero_grad()                
        output = model(x_tr)
        loss_tr = criterion(output,y_tr.view(-1))
        loss_tr.backward()
        optimizer.step()
        c.append((loss_tr.data[0]))           
    return c
        
def valid(x,y,n_valid_batch,num_c,model,cuda):
        
    index = np.random.permutation(len(x)) 
    model.eval()

    val= []
    for index_test in range(n_valid_batch): 
        v = []            
        x_va = x[index[index_test]]
        y_va = y[index[index_test]]        
        if cuda == True:
            x_va = Variable(x_va.float(),volatile=True).cuda() 
        else:
            x_va = Variable(x_va.float(),volatile=True) 
                                                                      
        y_va = y_va.view(-1,x_va.size(2)*x_va.size(3))
        if cuda == True:
            y_va = Variable(y_va.long(),volatile=True).cuda()
        else:
            y_va = Variable(y_va.long(),volatile=True)

        v.append(accuracy(x_va,y_va[0],model).data[0])
        for i in range(num_c):
            v.append(IoU(x_va,y_va[0],i,model).data[0])

        v.append(np.mean(v[1:]))
        val.append(v)
    return val
    
def test(x,y,n_test_batch,num_c,model,cuda):
        
    index = np.random.permutation(len(x)) 
    model.eval()
    loss = []
    for index_test in range(n_test_batch): 
        l = []            
        x_te = x[index[index_test]]
        y_te = y[index[index_test]]  
        if cuda == True:
            x_te = Variable(x_te.float(),volatile=True).cuda() 
        else:
            x_te = Variable(x_te.float(),volatile=True) 
                                                                          
        y_te = y_te.view(-1,x_te.size(2)*x_te.size(3))
        if cuda == True:
            y_te = Variable(y_te.long(),volatile=True).cuda()
        else:
            y_te = Variable(y_te.long(),volatile=True)
        l.append(accuracy(x_te,y_te[0],model).data[0])
        for i in range(num_c):
            l.append(IoU(x_te,y_te[0],i,model).data[0])

        l.append(np.mean(l[1:]))
        loss.append(l)
    return loss

    

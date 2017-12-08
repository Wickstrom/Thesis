import torch
import numpy as np
import torch.nn as nn
from FCDensenet import FCDensenet
from utils import train,valid,test
from pycrayon import CrayonClient


cc = CrayonClient(hostname="localhost")
data = torch.load('polyp_data.pth')

f = "FCDensenet.pth"
FCDensenet_experiment = cc.create_experiment("FCDensenet_experiment")

batch_size = 2
n_tr_batch = len(data[0]) // batch_size
n_va_batch = len(data[2]) // batch_size
n_te_batch = len(data[4]) // batch_size

n_epochs = 500
n_c = 2
k = 16
cuda = True

if cuda == True:
    model = FCDensenet(n_c,k,nn.ReLU(inplace=True)).cuda()
else:
    model = FCDensenet(n_c,k,nn.ReLU(inplace=True))
    
optimizer = torch.optim.RMSprop(model.parameters())  
criterion = nn.CrossEntropyLoss()

cost = []
va = []
te = []

patience = 0
best_valid = -100
tolerance = 50
start_transforming = 200
t_f = [(224,224),60,0.4,(0.7,1.4),False]

print('Training started')

for epoch in range(n_epochs):
    
    if epoch > start_transforming:
        t_f[4] = True 
    else:
        t_f[4] = False 
    
    if epoch == start_transforming:
        patience = 0
        tolerance = 75
       
    c = train(data[0],data[1],batch_size-1,n_tr_batch,optimizer,criterion,model,
                 t_f[0],t_f[1],t_f[2],t_f[3],t_f[4],cuda)
    v = valid(data[2],data[3],n_va_batch,n_c,model,cuda)                          
    l = test(data[4],data[5],n_te_batch,n_c,model,cuda)

    patience += 1
        
    FCDensenet_experiment.add_scalar_value("Cost",np.mean(c,0,dtype='float64'))
    
    FCDensenet_experiment.add_scalar_value("Mean Accuracy Validation",np.mean(v,0,dtype='float64')[0])
    FCDensenet_experiment.add_scalar_value("Background IoU Validation",np.mean(v,0,dtype='float64')[1])
    FCDensenet_experiment.add_scalar_value("Polyp IoU Validation",np.mean(v,0,dtype='float64')[2])
    FCDensenet_experiment.add_scalar_value("Mean IoU Validation",np.mean(v,0,dtype='float64')[3])
    
    FCDensenet_experiment.add_scalar_value("Mean Accuracy Test",np.mean(l,0,dtype='float64')[0])
    FCDensenet_experiment.add_scalar_value("Background IoU Test",np.mean(l,0,dtype='float64')[1])
    FCDensenet_experiment.add_scalar_value("Polyp IoU Test",np.mean(l,0,dtype='float64')[2])
    FCDensenet_experiment.add_scalar_value("Mean IoU Test",np.mean(l,0,dtype='float64')[3])

    if np.mean(v,0,dtype='float64')[2] > best_valid and epoch > 10 :
        best_valid = np.mean(v,0,dtype='float64')[2]
        torch.save(model,f)
        patience = 0        

    if np.mean(v,0,dtype='float64')[2] < best_valid and patience > tolerance:
        torch.save(model,f)
        print(" ######################################### ")
        print(" Break loop at iteration %d" % epoch)
        print(" Best validation score", best_valid)
        print(" Test score for this model", np.mean(l,0,dtype='float64'),)
        print(" ######################################### ")
        break




import torch
import numpy as np
import torch.nn as nn
#from fcn8 import FCN8
from segnet import SegNet
from datetime import datetime
from torch.autograd import Variable

data = torch.load('polyp_data.pth')

#f = "FCN8_net.pth"
f = "segnet_net.pth"
#g = "poly_graphs_fcn8.npz"
g = "poly_graphs_seg.npz"

batch_size = 5
n_train_batch = len(data[0]) // batch_size
n_valid_batch = len(data[2]) // batch_size
n_test_batch = len(data[4]) // batch_size

n_epochs = 200
weights = Variable(torch.Tensor((0.9,10))).data.cuda()

#model = FCN8(2).cuda()
model = SegNet(2).cuda()
optimizer = torch.optim.RMSprop(model.parameters(),weight_decay = 5 ** -4)  
criterion = nn.CrossEntropyLoss(weight=weights)

cost = []
valid = []
test = []
start = datetime.now()

print('Training started at:')
print(str(datetime.now()))

for epoch in range(n_epochs):
       
    c = model.tr(data[0],data[1],batch_size,n_train_batch,optimizer,criterion,model)
    v = model.va(data[2],data[3],n_valid_batch,model)                            
    l = model.te(data[4],data[5],n_test_batch,model)
   
    elapsed_time = datetime.now() - start
    
    cost.append(np.mean(c,0,dtype='float64'))
    valid.append(np.mean(v,0,dtype='float64'))
    test.append(np.mean(l,0,dtype='float64'))
    
    print("Training epoch %d" % epoch)
    print("Total elsapsed time is:")
    print(str(elapsed_time))    
    print("Training", cost[epoch])
    print("Validation", valid[epoch])
    print("Test", test[epoch])
    
    if epoch == 50 or epoch == 100 or epoch == 150:    
        torch.save(model,f)
               
torch.save(model,f)

np.savez_compressed(g,a=cost,b=valid,c=test)

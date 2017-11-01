import torch
import numpy as np
from fcn8_gb import FCN8_gb
from segnet_gb import SegNet_gb
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad

f =  'segnet_net.pth'  #'FCN8_net.pth' #'Unet_net.pth'
data = torch.load('polyp_data.pth')
model = torch.load(f,map_location=lambda storage, loc: storage)
#model_gb = FCN8_gb(2).cpu()
#model_gb = SegNet_gb(2).cpu()


model.eval()
X = Variable(data[4][0],requires_grad=True) 
out_large = np.argmax(model(X).data.numpy(),1).reshape(1,1,576,512)
x_large = data[4][0].numpy()

for i in range(58):
    model.eval()
    print(i)
    X = Variable(data[4][i+1],requires_grad=True)     
    out_temp = np.argmax(model(X).data.numpy(),1).reshape(1,1,576,512)
    out_large = np.concatenate((out_large,out_temp))
    print(out_large.shape)

X = Variable(data[4][59],requires_grad=True) 
out_small = np.argmax(model(X).data.numpy(),1).reshape(1,1,384,288)

for i in range(122):
    model.eval()
    print(i)
    X = Variable(data[4][i+59],requires_grad=True)     
    out_temp = np.argmax(model(X).data.numpy(),1).reshape(1,1,384,288)
    out_small = np.concatenate((out_small,out_temp))
    print(out_small.shape)

np.savez_compressed('polyp_preds_segnet.npz',a=out_large,b=out_small)

#x = data[4][0]
#for i in range(29):
#    x = torch.cat([x,data[4][i+150]])
# 
#X = Variable(x,requires_grad=True) 
#out = model_gb(X)
#output = F.softmax(out)
#
#one_hot = np.zeros((110592*30,2),dtype=np.float32)
#one_hot[np.arange(0,110592*30,1),np.argmax(output.data.numpy(),1)] = 1
#
#one_hot = Variable(torch.from_numpy(one_hot),requires_grad=True)
#one_hot = torch.sum(one_hot*out)
#out_ny = grad(one_hot,X)[0].data.numpy()
#X_out = out_ny

##x = data[4][59]
##for i in range(39):
##    x = torch.cat([x,data[4][i+60]])
## 
##X = Variable(x,requires_grad=True) 
##out = model_gb(X)
##output = F.softmax(out)
##
##one_hot = np.zeros((110592*40,2),dtype=np.float32)
##one_hot[np.arange(0,110592*40,1),np.argmax(output.data.numpy(),1)] = 1
##
##one_hot = Variable(torch.from_numpy(one_hot),requires_grad=True)
##one_hot = torch.sum(one_hot*out)
##out_ny = grad(one_hot,X)[0].data.numpy()
##X_out_s = out_ny
#
#np.savez_compressed('polyp_visualization_fcn8.npz',a=X_out)


#model.train()
#N = 59+46
#x1 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x2 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x3 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x4 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x5 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x6 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x7 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x8 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x9 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#x10 = F.softmax(model(Variable(data[4][N]))).data.numpy()
#print(x10.shape)
#
#n = 110592
#x = np.concatenate((x1.reshape(1,n,2),
#                    x2.reshape(1,n,2),
#                    x3.reshape(1,n,2),
#                    x4.reshape(1,n,2),
#                    x5.reshape(1,n,2),
#                    x6.reshape(1,n,2),
#                    x7.reshape(1,n,2),
#                    x8.reshape(1,n,2),
#                    x9.reshape(1,n,2),
#                    x10.reshape(1,n,2)))
#        
#
#np.savez_compressed('polyp_uncertainty_fcn3.npz',a=x)

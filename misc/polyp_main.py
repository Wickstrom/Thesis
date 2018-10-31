import torch
import pandas
import datetime
import numpy as np
import torch.nn as nn
from Unet import Unet
# from fcn8 import FCN8
# from segnet import SegNet

from utils import train, valid, test

f =   'Unet_net_ny.pth' # 'FCN8_net.pth' 'Segnet_net.pth'
g = 'Unet_graph_ny.npz' # 'FCN8_graph.npz' # 'Segnet_graph.npz' 
time_start = datetime.datetime.now()
data = torch.load('polyp_data.pth')

batch_size = 10
n_tr_batch = len(data[0]) // batch_size
n_va_batch, n_te_batch = 183, 182

n_epochs, n_c, cuda = 500, 2, True

if cuda:
    model = FCN8(n_c).cuda()
    # model = SegNet(n_c, nn.ReLU()).cuda()
    # model = Unet(n_c, nn.ReLU()).cuda()
else:
    model = FCN8(n_c)
    # model = SegNet(n_c, nn.ReLU())
    # model = Unet(n_c, nn.ReLU())

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

cost, validation, score = [], [], []

patience, tolerance = 0, 50
best_valid, best_test = -100, -100
start_transforming = 100
t_f = [(224, 224), 60, 50, (0.7, 1.4), False]

print('Training started at:')
print(time_start)
print('Model is:')
print(f)

for epoch in range(n_epochs):

    if epoch > start_transforming:
        t_f[4] = True
    else:
        t_f[4] = False

    if epoch == start_transforming:
        patience = 0

    c = train(data[0],
              data[1],
              batch_size-1,
              n_tr_batch,
              optimizer,
              criterion,
              model,
              t_f[0], t_f[1], t_f[2], t_f[3], t_f[4], cuda)

    v = valid(data[2],
              data[3],
              n_va_batch,
              n_c,
              model,
              cuda)

    loss = test(data[4],
                data[5],
                n_te_batch,
                n_c,
                model,
                cuda)

    patience += 1

    cost.append(np.mean(c, 0, dtype='float64'))
    validation.append(np.mean(v, 0, dtype='float64'))
    score.append(np.mean(loss, 0, dtype='float64'))

    print(pandas.DataFrame([epoch,
                            np.mean(c, 0, dtype='float64'),
                            datetime.datetime.now()-time_start],
                           ['Iteration', 'Cost', 'Elapsed time'],
                           [f]))

    print(pandas.DataFrame([np.mean(v, 0, dtype='float64')[0],
                            np.mean(v, 0, dtype='float64')[1],
                            np.mean(v, 0, dtype='float64')[2],
                            np.mean(v, 0, dtype='float64')[3]],
                           ['Mean Accuracy Validation',
                            'Background IoU Validation',
                            'Polyp IoU Validation',
                            'Mean IoU Validation'],
                           [f]))

    print(pandas.DataFrame([np.mean(loss, 0, dtype='float64')[0],
                            np.mean(loss, 0, dtype='float64')[1],
                            np.mean(loss, 0, dtype='float64')[2],
                            np.mean(loss, 0, dtype='float64')[3]],
                           ['Mean Accuracy Test',
                            'Background IoU Test',
                            'Polyp IoU Test',
                            'Mean IoU Test'],
                           [f]))

    if np.mean(v, 0, dtype='float64')[2] > best_valid and epoch > 10:
        best_valid = np.mean(v, 0, dtype='float64')[2]
        best_test = np.mean(loss, 0, dtype='float64')
        torch.save(model, f)
        np.savez_compressed(g, a=cost, b=validation, c=score)
        patience = 0

        print(" ########################################################### ")
        print(" Improved model at iteration %d " % epoch)
        print(" Validation score for this model ", best_valid)
        print(" Test score for this model", best_test,)
        print(" ########################################################### ")

    if np.mean(v, 0, dtype='float64')[2] < best_valid and patience > tolerance:
        np.savez_compressed(g, a=cost, b=validation, c=score)
        print(" ######################################### ")
        print(" Break loop at iteration %d" % epoch)
        print(" Best validation score", best_valid)
        print(" Best test score", best_test)
        print(" ######################################### ")
        break

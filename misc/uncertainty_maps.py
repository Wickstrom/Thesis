import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("file")
parser.add_argument("n_maps")
parser.add_argument("n_classes")
parser.add_argument('--cuda', action='store_true', default=False)

args = parser.parse_args()
data = torch.load('polyp_data.pth')

if args.cuda:
    print('cuda')
    model = torch.load(args.model).cuda()
else:
    model = torch.load(args.model).cpu()
    print('cpu')

n_maps = np.int(args.n_maps)
n_classes = np.int(args.n_classes)
predictions = []

for i in range(182):
    print('Sample number:')
    print(i)
    col = np.int(data[4][i].size(2)*data[4][i].size(3))
    temp = np.zeros((n_maps, col, n_classes))
    for j in range(n_maps):
        model.train()

        if args.cuda:
            output = model(Variable(data[4][i]).float()).cuda()
        else:
            output = model(Variable(data[4][i]).float()).cpu()

        temp[j, :, :] = F.softmax(output).data.numpy()

    predictions.append(temp)

np.savez_compressed(args.file, a=predictions[:59], b=predictions[59:])

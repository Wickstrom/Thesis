import torch
import argparse
import numpy as np
from torch.autograd import grad
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("file")
parser.add_argument('--cuda', action='store_true', default=False)

args = parser.parse_args()
data = torch.load('polyp_data.pth')

if args.cuda:
    print('cuda')
    model = torch.load(args.model).cuda()
else:
    model = torch.load(args.model).cpu()
    print('cpu')

gradients = []

for i in range(182):
    print('Sample number:')
    print(i)
    model.eval()
    temp = Variable(data[4][i], requires_grad=True).float()
    X = temp.repeat(5, 1, 1, 1)
    N_samples = np.int(data[4][i].size(2)*data[4][i].size(3)) * 5
    one_hot = np.zeros((N_samples, 2), dtype=np.float32)

    if args.cuda:
        output = model(X).cuda()
    else:
        output = model(X).cpu()

    pred = output.data.numpy()

    one_hot[np.arange(0, N_samples, 1), np.argmax(pred, 1)] = 1
    one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
    one_hot = torch.sum(one_hot*output)
    gradient = grad(one_hot, X)[0].data.numpy()
    gradients.append(gradient[0])
    model.zero_grad()

np.savez_compressed(args.file, a=gradients[:59], b=gradients[59:])

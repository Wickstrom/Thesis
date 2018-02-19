import torch
import argparse
import numpy as np
import torch.nn.functional as F
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

predictions = []

for i in range(182):
    print('Sample number:')
    print(i)
    model.eval()

    if args.cuda:
        output = model(Variable(data[4][i]).float()).cuda()
    else:
        output = model(Variable(data[4][i]).float()).cpu()

    predictions.append(np.argmax(F.softmax(output).data.numpy(), 1))

np.savez_compressed(args.file, a=predictions)

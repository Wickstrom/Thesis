# %%
import scipy
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist, squareform

# %%


def renyi(A):
    alpha = 2.0
    l, v = LA.eig(A)
    return (1/(1-alpha))*np.log2(np.sum(l**alpha))


def joint_renyi(A, B):
    AB = np.multiply(A, B)
    AB = AB / np.trace(AB)
    return renyi(AB)


def mutual_renyi(A, B):
    return renyi(A)+renyi(B)-joint_renyi(A, B)


def mutual_renyi_normalized(A, B):
    return mutual_renyi(A, B) / np.sqrt(renyi(A)*renyi(B))


def silverman(h, N):
    return h*N**(-1/(1+N))


data = np.load('polyp_data.npz')['a']
x_te1 = data[4]
prediction1 = np.load('MI_estimates.npz', encoding='latin1')['a']
enc31 = np.load('MI_estimates.npz', encoding='latin1')['b']
enc41 = np.load('MI_estimates.npz', encoding='latin1')['c']
enc51 = np.load('MI_estimates.npz', encoding='latin1')['d']
dec31 = np.load('MI_estimates.npz', encoding='latin1')['e']
dec41 = np.load('MI_estimates.npz', encoding='latin1')['f']
dec51 = np.load('MI_estimates.npz', encoding='latin1')['g']

# %%

N = 100

idx = np.random.choice(np.arange(59, 181, 1), size=(N), replace=False)

a, b, c = x_te1[idx[0]].shape
ixe2, ixe3, ixe4, ixe5 = b//4, b//8, b//16, b//32
iye2, iye3, iye4, iye5 = c//4, c//8, c//16, c//32

x_sample = x_te1[idx[0]].reshape(1, 3, b, c) / 255
y_sample = prediction1[idx[0]][:, 0].reshape(1, 1, b, c)

enc3_sample = np.transpose(enc31[idx[0]]).reshape(1, 256, ixe3, iye3)
enc4_sample = np.transpose(enc41[idx[0]]).reshape(1, 512, ixe4, iye4)
enc5_sample = np.transpose(enc51[idx[0]]).reshape(1, 512, ixe5, iye5)

dec3_sample = np.transpose(dec31[idx[0]]).reshape(1, 128, ixe2, iye2)
dec4_sample = np.transpose(dec41[idx[0]]).reshape(1, 256, ixe3, iye3)
dec5_sample = np.transpose(dec51[idx[0]]).reshape(1, 512, ixe4, iye4)


for i in range(N-1):

    x_sample = np.concatenate((x_sample,
                               x_te1[idx[i+1]].reshape(1, 3, 384, 288) / 255))

    y_sample = np.concatenate((y_sample,
                              prediction1[idx[i+1]][:, 0].reshape(1, 1, b, c)))

    enc3_sample = np.concatenate((enc3_sample,
                                  np.transpose(enc31[idx[i+1]]).reshape(1, 256, ixe3, iye3)))

    enc4_sample = np.concatenate((enc4_sample,
                                  np.transpose(enc41[idx[i+1]]).reshape(1, 512, ixe4, iye4)))

    enc5_sample = np.concatenate((enc5_sample,
                                  np.transpose(enc51[idx[i+1]]).reshape(1, 512, ixe5, iye5)))

    dec3_sample = np.concatenate((dec3_sample,
                                  np.transpose(dec31[idx[i+1]]).reshape(1, 128, ixe2, iye2)))

    dec4_sample = np.concatenate((dec4_sample,
                                  np.transpose(dec41[idx[i+1]]).reshape(1, 256, ixe3, iye3)))

    dec5_sample = np.concatenate((dec5_sample,
                                  np.transpose(dec51[idx[i+1]]).reshape(1, 512, ixe4, iye4)))


x_sample = x_sample.reshape(N, 3*b*c)
y_sample = y_sample.reshape(N, 1*b*c) / np.max(y_sample)

enc3_sample = enc3_sample.reshape(N, 256*ixe3*iye3) / np.max(enc3_sample)
enc4_sample = enc4_sample.reshape(N, 512*ixe4*iye4) / np.max(enc4_sample)
enc5_sample = enc5_sample.reshape(N, 512*ixe5*iye5) / np.max(enc5_sample)

dec3_sample = dec3_sample.reshape(N, 128*ixe2*iye2) / np.max(dec3_sample)
dec4_sample = dec4_sample.reshape(N, 256*ixe3*iye3) / np.max(dec4_sample)
dec5_sample = dec5_sample.reshape(N, 512*ixe4*iye4) / np.max(dec5_sample)

# %%

all_k = np.zeros((8, N, N))
M = 10

sigma_x = np.mean(np.mean(np.sort(
                    squareform(pdist(x_sample, 'euclidean')))[:, :M], 1))
sigma_y = np.mean(np.mean(np.sort(
                    squareform(pdist(y_sample, 'euclidean')))[:, :M], 1))
sigma_enc3 = np.mean(np.mean(np.sort(
                    squareform(pdist(enc3_sample, 'euclidean')))[:, :M], 1))
sigma_enc4 = np.mean(np.mean(np.sort(
                    squareform(pdist(enc4_sample, 'euclidean')))[:, :M], 1))
sigma_enc5 = np.mean(np.mean(np.sort(
                    squareform(pdist(enc5_sample, 'euclidean')))[:, :M], 1))
sigma_dec3 = np.mean(np.mean(np.sort(
                    squareform(pdist(dec3_sample, 'euclidean')))[:, :M], 1))
sigma_dec4 = np.mean(np.mean(np.sort(
                    squareform(pdist(dec4_sample, 'euclidean')))[:, :M], 1))
sigma_dec5 = np.mean(np.mean(np.sort(
                    squareform(pdist(dec5_sample, 'euclidean')))[:, :M], 1))

silverman_sigma = silverman(5, N)
# sigma_x = sigma_y = sigma_enc3 = sigma_enc4 = sigma_enc5 = sigma_dec3 = sigma_dec4 = sigma_dec5 = silverman(5, N) 

x_k = (1/(N))*(scipy.exp(-squareform(pdist(x_sample,
                                           'euclidean')) ** 2 / sigma_x ** 2))
y_k = (1/(N))*(scipy.exp(-squareform(pdist(y_sample,
                                           'euclidean')) ** 2 / sigma_y ** 2))

enc3_k = (1/(N))*(scipy.exp(-squareform(pdist(enc3_sample,
                  'euclidean')) ** 2 / sigma_enc3 ** 2))
enc4_k = (1/(N))*(scipy.exp(-squareform(pdist(enc4_sample,
                  'euclidean')) ** 2 / sigma_enc4 ** 2))
enc5_k = (1/(N))*(scipy.exp(-squareform(pdist(enc5_sample,
                  'euclidean')) ** 2 / sigma_enc5 ** 2))

dec3_k = (1/(N))*(scipy.exp(-squareform(pdist(dec3_sample,
                  'euclidean')) ** 2 / sigma_dec3 ** 2))
dec4_k = (1/(N))*(scipy.exp(-squareform(pdist(dec4_sample,
                  'euclidean')) ** 2 / sigma_dec4 ** 2))
dec5_k = (1/(N))*(scipy.exp(-squareform(pdist(dec5_sample,
                  'euclidean')) ** 2 / sigma_dec5 ** 2))


all_k[0, :, :], all_k[1, :, :], all_k[2, :, :] = x_k, y_k, enc3_k
all_k[3, :, :], all_k[4, :, :], all_k[5, :, :] = enc4_k, enc5_k, dec3_k
all_k[6, :, :], all_k[7, :, :] = dec4_k, dec5_k

# %%

all_MI = np.zeros((8, 8))

for i in range(8):
    for j in range(8):
        all_MI[i, j] = mutual_renyi(all_k[i, :, :], all_k[j, :, :])

print(all_MI)

all_MI_normalized = np.zeros((8, 8))

for i in range(8):
    for j in range(8):
        all_MI_normalized[i, j] = mutual_renyi_normalized(
                    all_k[i, :, :], all_k[j, :, :])

print(all_MI_normalized)



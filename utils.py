import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
import PIL.Image as Image

DEVICE = 'cuda'
mse = nn.MSELoss()


def calc_histogram_loss(A, B, histogram_block):
    input_hist = histogram_block(A)
    target_hist = histogram_block(B)
    histogram_loss = (1/np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) / 
        input_hist.shape[0])

    return histogram_loss
    
# B, C, H, W; mean var on HW
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def train_transform():
    transform_list = [
        transforms.Resize(size=512),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize(size=(512)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

# https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            print('-'*82)
            print(n, p.grad.abs().mean(), p.grad.abs().max())
            print('-'*82)

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count, args):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + 5e-5 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cosine_dismat(A, B):
    A = A.view(A.shape[0], A.shape[1], -1)
    B = B.view(B.shape[0], B.shape[1], -1)

    A_norm = torch.sqrt((A**2).sum(1))
    B_norm = torch.sqrt((B**2).sum(1))

    A = (A/A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0,2,1)
    B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))
    dismat = 1.-torch.bmm(A, B) 

    return dismat

def calc_remd_loss(A, B):
    C = cosine_dismat(A, B)
    m1, _ = C.min(1)
    m2, _ = C.min(2)
    
    remd = torch.max(m1.mean(), m2.mean())

    return remd

def calc_ss_loss(A, B):
    MA = cosine_dismat(A, A)
    MB = cosine_dismat(B, B)
    Lself_similarity = torch.abs(MA-MB).mean() 

    return Lself_similarity

def calc_moment_loss(A, B):
    A = A.view(A.shape[0], A.shape[1], -1)
    B = B.view(B.shape[0], B.shape[1], -1)

    mu_a = torch.mean(A, 1, keepdim=True)
    mu_b = torch.mean(B, 1, keepdim=True)
    mu_d = torch.abs(mu_a - mu_b).mean()

    A_c = A - mu_a
    B_c = B - mu_b
    cov_a = torch.bmm(A_c, A_c.permute(0,2,1)) / (A.shape[2]-1)
    cov_b = torch.bmm(B_c, B_c.permute(0,2,1)) / (B.shape[2]-1)
    cov_d = torch.abs(cov_a - cov_b).mean()
    loss = mu_d + cov_d
    return loss

def calc_mse_loss(A, B):
    return mse(A, B)


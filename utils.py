import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys
import random
from torchvision.transforms.functional import normalize

def get_dataset_mean(training_generator):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in training_generator:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

def batch_intersection_union(predict, target, num_class):
    _, predict = torch.max(predict, 1)
    if target.shape == torch.Size([4, 3, 256, 256]):
        _, target = torch.max(target,1)


    predict = predict + 1 #no intersection indexes will be 0. Overlaps with background, thus, as a solution, is why all indexes are incrased by 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count
            
def write_logger(filename_log, cfg, **kwargs):
    if not os.path.isdir('results'):
        os.mkdir('results')
    f = open('results/'+filename_log, "a")
    if kwargs['epoch'] == 0:
        f.write("Training CONFIGS are: "+ 
        "SRM="+str(cfg['global_params']['with_srm'])+ " "+ "Contrastive="+ str(cfg['global_params']['with_con']) + " " 
        +"Encoder Name: "+cfg['model_params']['encoder'] + "\n" )
        f.write("\n")
    

    for key, value in kwargs.items():
        f.write(str(key) +": " +str(value)+ "\n")
    f.write("\n")
    f.close()

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def calfeaturevectors(feat, mask):
    #feat and mask both should be BXCXHXW
    out = torch.mul(feat.unsqueeze(dim=1), mask.unsqueeze(dim=2)).float()
    sum_feat = torch.sum(out, dim=(3,4))
    mask_sum = torch.sum(mask, dim=(2,3))
    mean_feat = sum_feat/((mask_sum+1e-16).unsqueeze(dim=2))
    return mean_feat

def one_hot(label, n_class, device = None):
    #label should be BXHXW
    B,H,W = label.shape
    encoded = torch.zeros(size=(B,n_class,H,W)) 
    if device:
        encoded = encoded.to(device)
    encoded = encoded.scatter_(1, label.unsqueeze(1), 1)
    return encoded


def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function, Variable

def get_con_loss(feat,img,tar,device):
    feat = F.interpolate(feat, img.shape[2:], mode='bilinear', align_corners=True)
    cfeature = F.avg_pool2d(feat, 8, 8) 
    cfeature = (cfeature > 0.5).int().float()
    Ba, Ch, _, _ = cfeature.shape
    cfeature = cfeature.view(Ba, Ch, -1)
    cfeature = torch.transpose(cfeature, 1, 2)  
    cfeature = F.normalize(cfeature, dim=-1)

    mask_con = tar.detach()
    mask_con = mask_con.float()
    mask_con = F.avg_pool2d(mask_con, 8, 8)
    mask_con = (mask_con > 0.5).int().float()

    mask_con = mask_con.view(Ba, -1)  
    mask_con = mask_con.unsqueeze(dim=1)

    contrast_temperature = 0.1
    c_loss = square_patch_contrast_loss(cfeature, mask_con, device, contrast_temperature)
    c_loss = c_loss.mean(dim=-1)
    c_loss = c_loss.mean()
    return c_loss
def square_patch_contrast_loss(feat, mask, device, temperature=0.6):
    mem_mask = torch.eq(mask, mask.transpose(1, 2)).float().to(device)
    mem_mask_neg = torch.add(torch.negative(mem_mask), 1).to(device)
    feat_logits = torch.div(torch.matmul(feat, feat.transpose(1, 2)), temperature).to(device)
    identity = torch.eye(feat_logits.shape[-1]).to(device)
    neg_identity = torch.add(torch.negative(identity), 1).detach()

    feat_logits = torch.mul(feat_logits, neg_identity)

    feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
    feat_logits = feat_logits - feat_logits_max.detach()

    feat_logits = torch.exp(feat_logits)

    neg_sum = torch.sum(torch.mul(feat_logits, mem_mask_neg), dim=-1)
    denominator = torch.add(feat_logits, neg_sum.unsqueeze(dim=-1))

    division = torch.div(feat_logits, denominator + 1e-18)

    loss_matrix = -torch.log(division + 1e-18)
    loss_matrix = torch.mul(loss_matrix, mem_mask)
    loss_matrix = torch.mul(loss_matrix, neg_identity)
    loss = torch.sum(loss_matrix, dim=-1)

    loss = torch.div(loss, mem_mask.sum(dim=-1) - 1 + 1e-18)

    return loss


def computeF1(FP, TP, FN, TN):
    return 2 * TP / np.maximum((2 * TP + FN + FP), 1e-32)


def computeMetrics_th(values, gt, gt0, gt1, th):
    values = values > th
    values = values.flatten().astype(np.uint8)
    gt = gt.flatten().astype(np.uint8)
    gt0 = gt0.flatten().astype(np.uint8)
    gt1 = gt1.flatten().astype(np.uint8)

    gt = gt[(gt0 + gt1) > 0]
    values = values[(gt0 + gt1) > 0]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt, values, labels=[0, 1])

    TN = cm[0, 0]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TP = cm[1, 1]

    return FP, TP, FN, TN

def extractGTs(gt, erodeKernSize=15, dilateKernSize=11):
    from scipy.ndimage import minimum_filter, maximum_filter
    gt1 = minimum_filter(gt, erodeKernSize)
    gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
    return gt0, gt1


def computeMetricsContinue(values, gt0, gt1):
    values = values.flatten().astype(np.float32)
    gt0 = gt0.flatten().astype(np.float32)
    gt1 = gt1.flatten().astype(np.float32)

    inds = np.argsort(values)
    inds = inds[(gt0[inds] + gt1[inds]) > 0]
    vet_th = values[inds]
    gt0 = gt0[inds]
    gt1 = gt1[inds]

    TN = np.cumsum(gt0)
    FN = np.cumsum(gt1)
    FP = np.sum(gt0) - TN
    TP = np.sum(gt1) - FN

    msk = np.pad(vet_th[1:] > vet_th[:-1], (0, 1), mode='constant', constant_values=True)
    FP = FP[msk]
    TP = TP[msk]
    FN = FN[msk]
    TN = TN[msk]
    vet_th = vet_th[msk]

    return FP, TP, FN, TN, vet_th
def computeLocalizationMetrics(map, gt):
    gt0, gt1 = extractGTs(gt)

    # best threshold
    try:
        FP, TP, FN, TN, _ = computeMetricsContinue(map, gt0, gt1)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_best = max(np.max(f1), np.max(f1i))
    except:
        import traceback
        traceback.print_exc()
        F1_best = np.nan

    # fixed threshold
    try:
        FP, TP, FN, TN = computeMetrics_th(map, gt, gt0, gt1, 0.5)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_th = max(f1, f1i)
    except:
        import traceback
        traceback.print_exc()
        F1_th = np.nan

    return F1_best, F1_th


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
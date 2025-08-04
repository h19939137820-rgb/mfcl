import torch
import os

from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from MFCL.mfcl import get_MFCL
from MFCL.sedge import get_sobel, run_sobel
from metrics import get_con_loss, computeLocalizationMetrics
from utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
from sklearn import metrics
import datetime
import numpy as np
import yaml
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

tb = SummaryWriter()
testi = 1
set_random_seed(1221)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


now = datetime.datetime.now()
filename_log = 'Results-' + str(now) + '.txt'


checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
img_size = 384

# train_dataset, test_dataset = random_split(tamp_dataset, [train_size, len(tamp_dataset) - train_size])
#
# training_generator = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
# validation_generator = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=8)
# training_generator = DataLoader(tamp_dataset, batch_size=8, shuffle=True, num_workers=8)


model = get_MFCL().to(device)

optimizer = optim.Adam(model.parameters(),lr=0.0001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
imbalance_weight = torch.tensor([0.0892, 0.9108]).to(device)
criterion = nn.CrossEntropyLoss(weight=imbalance_weight)
#
checkpoint_path = os.path.join('checkpoints', 'inceptionnext_384_checkpoint_rectify.pth')

if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    try:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        print("Model weights loaded successfully with partial matching.")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(e, "error")
else:
    print(f"Checkpoint file '{checkpoint_path}' not found. Training from scratch.")

# 评价指标
max_val_auc = 0
max_val_iou = [0.0, 0.0]
max_f1 = 0


for epoch in range(30):
    for sample in tqdm(training_generator):
        model.train()
        optimizer.zero_grad()

        img = sample[0].to(device)
        tar = sample[1].to(device)

        edge = tar.unsqueeze(1)
        sobel_x, sobel_y = get_sobel(1, 1)
        sobel_x, sobel_y = sobel_x.to(device), sobel_y.to(device)
        edge_labels = run_sobel(sobel_x, sobel_y, edge)
        edge_labels = edge_labels.squeeze(1)

        pred, feat, edge = model(img)

        pred = F.interpolate(pred, img.shape[2:], mode='bilinear', align_corners=True)
        feat = F.interpolate(feat, img.shape[2:], mode='bilinear', align_corners=True)
        edge = F.interpolate(edge, img.shape[2:], mode='bilinear', align_corners=True)

        c_loss = get_con_loss(feat,img,tar,device)
        loss = criterion(pred, tar.long().detach())
        edge_loss = criterion(edge, edge_labels.long().detach())
        total_loss = loss + 0.1 * c_loss + edge_loss
        total_loss.backward()
        optimizer.step()

    scheduler.step()
    # with torch.no_grad():
    #     model.eval()
    #     val_inter = AverageMeter()
    #     val_union = AverageMeter()
    #     val_pred = []
    #     val_tar = []
    #     auc = []
    #     f1 = []
    #     f1th = []
    #     for img, tar in tqdm(validation_generator):
    #         img, tar = img.to(device), tar.to(device)
    #         pred, _, edge = model(img)  # 4 256 256
    #         pred = F.interpolate(pred, img.shape[2:], mode='bilinear', align_corners=True)
    #         intr, uni = batch_intersection_union(pred, tar, num_class=1)
    #         val_inter.update(intr)
    #         val_union.update(uni)
    #         y_score = F.softmax(pred, dim=1)[:, 1, :, :]
    # 
    #         for yy_true, yy_pred in zip(tar.cpu().numpy(), y_score.cpu().numpy()):
    #             try:
    #                 this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel())
    #                 that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1 - yy_pred).ravel())
    #                 auc.append(max(this, that))
    # 
    #                 F1_best, F1_th = computeLocalizationMetrics(yy_pred, yy_true)
    #                 f1.append(F1_best)
    #                 f1th.append(F1_th)
    #             except ValueError:
    #                 pass
    #     val_auc = np.mean(auc)
    #     mf1 = np.mean(f1)
    #     midf1 = np.mean(f1th)
    #     print(f"auc:{val_auc},f1:{mf1},f1th:{midf1}")
    #     val_pred = []
    #     val_tar = []
    #     if val_auc > max_val_auc or mf1 > max_f1:
    #         max_val_auc = val_auc
    #         max_f1 = mf1
    #         checkpoint_path = os.path.join(checkpoint_dir, f'inceptionnext_384_checkpoint_rectify_{testi}.pth')
    # 
    #         torch.save({
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #         }, checkpoint_path)
    #         testi += 1
    # 
    #         logs = {'epoch': epoch, 'Validation AUC': val_auc,
    #                     'Max Validaton_AUC': max_val_auc, "Max IoU Tampered": max_val_iou, "max_f1": mf1,
    #                     "mean_f1": midf1}
    #         tb.add_scalar("auc", val_auc, epoch + 1)
    #         write_logger(filename_log, **logs)
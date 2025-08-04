import random
import imageio
from PIL import Image



# class tampCOCO(Dataset):
#     def __init__(self, args):
#         super(tampCOCO, self).__init__()
#         self.crop_size = args['crop_size']
#         self.path = args['path']
#         self.bcm_list = []
#         self.cm_list = []
#         self.sp_list = []
#         self.tamp_list = []
#         file_names = ['bcm_COCO_list.txt', 'cm_COCO_list.txt', 'sp_COCO_list.txt']
#         lines_to_read = 10000  # 指定要读取的行数
#
#         for file_name in file_names:
#             with open(os.path.join(self.path, file_name), "r") as f:
#                 lines_read = 0
#                 for line in f:
#                     self.tamp_list.append(line.strip().split(','))
#                     lines_read += 1
#                     if lines_read >= lines_to_read:
#                         break
#
#     def __getitem__(self, index):
#         crop_width, crop_height = self.crop_size
#         image_name = self.path + self.tamp_list[index][0]
#         mask_name = self.path + self.tamp_list[index][1]



import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def data_aug(img, data_aug_ind):
    img = Image.fromarray(img)
    if data_aug_ind == 0:
        return np.asarray(img)
    elif data_aug_ind == 1:
        return np.asarray(img.rotate(90, expand=True))
    elif data_aug_ind == 2:
        return np.asarray(img.rotate(180, expand=True))
    elif data_aug_ind == 3:
        return np.asarray(img.rotate(270, expand=True))
    elif data_aug_ind == 4:
        return np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 5:
        return np.asarray(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 6:
        return np.asarray(img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 7:
        return np.asarray(img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    else:
        raise Exception('Data augmentation index is not applicable.')
class tampCOCO(Dataset):
    def __init__(self, args):
        super(tampCOCO, self).__init__()
        self.crop_size = args['crop_size']
        self.path = args['path']
        self.bcm_list = []
        self.cm_list = []
        self.sp_list = []
        self.tamp_list = []
        file_names = ['bcm_COCO_list.txt', 'cm_COCO_list.txt', 'sp_COCO_list.txt']
        self.imgsize = 384
        # first_line = 0
        # lines_to_read = 30000 # 指定要读取的行数
        # lines_to_read = 10000 # 指定要读取的行数
        for file_name in file_names:
            with open(os.path.join(self.path, file_name), "r") as f:
                lines_read = 0
                for line in f:
                    self.tamp_list.append(line.strip().split(','))
                    lines_read += 1
                    if lines_read >= lines_to_read:
                        break

    def __len__(self):
        return len(self.tamp_list)


    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')
    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size

        image_name = os.path.join(self.path, self.tamp_list[index][0])
        mask_name = os.path.join(self.path, self.tamp_list[index][1])
    
        tp_img = imageio.imread(image_name)
        if tp_img.shape[-1] == 4:
            tp_img = self.rgba2rgb(tp_img)
        mask_img = imageio.imread(mask_name, mode='L')


        # resize
        tp_img = Image.fromarray(tp_img)
        tp_img = tp_img.resize((self.imgsize, self.imgsize), resample=Image.BICUBIC)
        tp_img = np.asarray(tp_img)
        # resize mask
        mask_img = Image.fromarray(mask_img)
        mask_img = mask_img.resize((self.imgsize, self.imgsize), resample=Image.BICUBIC)
        mask_img = np.asarray(mask_img)

        aug_index = random.randrange(0, 8)
        tp_img = data_aug(tp_img, aug_index)
        tp_img = torch.from_numpy(tp_img.astype(np.float32) / 255).permute(2, 0, 1)
        mask_img = data_aug(mask_img, aug_index)
        mask_img = torch.from_numpy(mask_img.astype(np.float32) / 255)

        return tp_img, mask_img


if __name__ == '__main__':
    args = {"crop_size":(1024,512),"path":'/root/autodl-tmp/tampCOCO'}
    dataset = tampCOCO(args)
    print(len(dataset))
    # 加载第一个数据样本x`
    idx = 0
    image, mask = dataset[idx]
    print(image.shape,mask.shape)

    # # 可视化数据
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.moveaxis(image.numpy(), 0, 2))
    # plt.title('Image')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask, cmap='gray')
    # plt.title('Mask')
    #
    # plt.show()


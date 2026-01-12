# -*- coding: utf-8 -*-
import numpy as np
from tqdm import trange
import math
import os

import cv2

"""
图像分割任务数据预处理
输入img和gt图按照patch和overlap参数切块
"""


def slice_tiles(img_root, gt_root, slice_img_save_root, slice_gt_save_root, patch_size, overlap):
    img_files = os.listdir(img_root)
    gt_files = os.listdir(gt_root)
    bar = trange(len(img_files))
    for (img_file_, gt_file_, _) in zip(img_files, gt_files, bar):
        img_filename = img_root+img_file_
        gt_filename = gt_root+gt_file_
        img = cv2.imread(img_filename)
        gt = cv2.imread(gt_filename)

        rows, cols = img.shape[0], img.shape[1]
        # 计算横竖各有多少块
        m = math.ceil((rows - patch_size) / (patch_size - overlap) + 1)
        n = math.ceil((cols - patch_size) / (patch_size - overlap) + 1)

        k = 0
        for i in range(m):
            x1 = i * (patch_size - overlap)
            x2 = x1 + patch_size
            if x2 >= rows:  # 超出图像范围
                x1, x2 = rows - patch_size, rows
            if i == 0:
                x1, x2 = 0, patch_size

            for j in range(n):
                y1 = j * (patch_size - overlap)
                y2 = y1 + patch_size
                if y2 >= cols:  # 超出图像范围
                    y1, y2 = cols - patch_size, cols
                if j == 0:
                    y1, y2 = 0, patch_size

                img_patch = img[x1:x2, y1:y2, :]
                gt_patch = gt[x1:x2, y1:y2]

                k += 1
                cv2.imwrite(os.path.join(slice_img_save_root, img_file_.replace('.jpg', '_')+str(k)+'.jpg'), img_patch)
                cv2.imwrite(os.path.join(slice_gt_save_root, gt_file_.replace('.png', '_')+str(k)+'.png'), gt_patch)


if __name__ == '__main__':

    # 设置分割图像块大小
    patch_size = 1024
    # 设置分割块重叠度
    overlap = 400

    img_root = "G:/data_backup/CHAIR/data/connector2_defect/dataset/img/"
    gt_root = "G:/data_backup/CHAIR/data/connector2_defect/dataset/gt/"

    slice_img_save_root = "G:/data_backup/CHAIR/data/connector2_defect/dataset/img_cut/"
    slice_gt_save_root = "G:/data_backup/CHAIR/data/connector2_defect/dataset/gt_cut/"

    if not os.path.exists(slice_img_save_root):
        os.makedirs(slice_img_save_root)
    if not os.path.exists(slice_gt_save_root):
        os.makedirs(slice_gt_save_root)

    slice_tiles(img_root, gt_root, slice_img_save_root, slice_gt_save_root,patch_size,overlap)
    print("finished!")

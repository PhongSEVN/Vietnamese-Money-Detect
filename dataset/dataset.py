import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from config.mobile_train_config import MOBILE_DATA_DIR


class Money(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        if train:
            mode = 'train'
        else:
            mode = 'test'
        self.root = os.path.join(root, mode)
        self.transform = transform
        self.categories = sorted(os.listdir(self.root))
        self.image_path = []
        self.label_path = []

        for i, category in enumerate(self.categories):
            data_file_path = os.path.join(self.root, category)

            for filename in os.listdir(data_file_path):
                image_path = os.path.join(data_file_path, filename)
                self.image_path.append(image_path)
                self.label_path.append(i)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        label_path = self.label_path[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label_path

    def __len__(self):
        return len(self.label_path)

if __name__ == '__main__':
    dataset = Money(root=MOBILE_DATA_DIR, train=False, transform=None)
    print(dataset.categories)
    # print(dataset.image_path)
    # print(dataset.label_path)
    print(dataset.__len__())
    img, label = dataset.__getitem__(100)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (600, 600))
    print(cv2.imshow("image", img))
    cv2.waitKey(0)



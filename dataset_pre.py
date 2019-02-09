from torch.utils.data import Dataset
import torch
import cv2
import os
import numpy as np
class dataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = cv2.imread(img_name)
        box = np.array([self.df.iloc[idx, 1], self.df.iloc[idx, 3], self.df.iloc[idx, 2], self.df.iloc[idx, 4]])
        edges = cv2.Canny(image,100,200)
        kernel = np.ones((5,5), np.uint8)
        img_dilation = cv2.dilate(edges, kernel, iterations=1)
        img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
        img1 = cv2.medianBlur(img_erosion,3)
        img_dilation = cv2.dilate(img1, kernel, iterations=1)
        img2=img_dilation.reshape(1,640, 480)
        image = image.reshape(3,640,480)
        image = np.concatenate((image, img2), axis = 0)
        image = image.reshape(4,640,480)
        return {'image': torch.from_numpy(image).type(torch.DoubleTensor).double().cuda(),
                'box': torch.from_numpy(box)}
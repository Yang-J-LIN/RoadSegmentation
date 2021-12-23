import os

import numpy as np

import cv2 as cv

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms.functional import rotate
import torchvision.transforms.functional as Tf
import torchvision.transforms as T

from tqdm import tqdm


def load_image(path, color_space=cv.IMREAD_GRAYSCALE):
    img = cv.imread(path, color_space)

    if color_space == cv.IMREAD_GRAYSCALE:
        # img = cv.equalizeHist(img)
        pass
    else:
        # (b, g, r) = cv.split(img)
        # bH = cv.equalizeHist(b)
        # gH = cv.equalizeHist(g)
        # rH = cv.equalizeHist(r)
        # img = cv.merge((bH, gH, rH))
        pass

        # img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    return img


def aumgent_data(images, groundtruth):  # images: N * 3 * 400 * 400, groundtruth: N * 400 * 400

    images = (images * 255).to(torch.uint8)
    groundtruth = (groundtruth * 255).to(torch.uint8)

    images_augmented = images
    groundtruth_augmented = groundtruth

    # Five Crop images into smaller sizes.

    images_augmented = []
    groundtruth_augmented = []

    images_augmented = torch.cat(list(T.FiveCrop(size=(200,200))(images)), dim = 0)
    groundtruth_augmented  = torch.cat(list(T.FiveCrop(size=(200,200))(groundtruth)), dim = 0)

    images_augmented = torch.cat(list(T.FiveCrop(size=(128,128))(images_augmented)), dim = 0)
    groundtruth_augmented  = torch.cat(list(T.FiveCrop(size=(128,128))(groundtruth_augmented)), dim = 0)

    images_augmented_filtered = []
    groundtruth_augmented_filtered = []

    for i, j in zip(images_augmented, groundtruth_augmented):
        if (j > 0).sum() / (j.size()[-2] * j.size()[-1]) > 0.1:
            images_augmented_filtered.append(i.unsqueeze(0))
            groundtruth_augmented_filtered.append(j.unsqueeze(0))

    # images_augmented = torch.cat(images_augmented_filtered, dim=0)
    # groundtruth_augmented = torch.cat(groundtruth_augmented_filtered, dim=0)
    # print(images_augmented.shape, groundtruth_augmented.shape)
    # ------------------------------------------------------ #
    images_augmented_rotated = []
    groundtruth_augmented_rotated = []
    for degree in range(0, 360, 45):
        images_augmented_rotated = images_augmented_rotated + [T.functional.rotate(images_augmented,degree)]
        groundtruth_augmented_rotated  = groundtruth_augmented_rotated + [T.functional.rotate(groundtruth_augmented, degree)]
    images_augmented = torch.cat(images_augmented_rotated, dim=0)
    groundtruth_augmented = torch.cat(groundtruth_augmented_rotated, dim=0)

    print(images_augmented.shape, groundtruth_augmented.shape)
    
    # # 2. Flips 
    # images_augmented = [images_augmented, T.RandomHorizontalFlip(p=1.0)(images_augmented),T.RandomVerticalFlip(p=1.0)(images_augmented)]
    # groundtruth_augmented = [groundtruth_augmented, T.RandomHorizontalFlip(p=1.0)(groundtruth_augmented),T.RandomVerticalFlip(p=1.0)(groundtruth_augmented)]
    # images_augmented = torch.cat(images_augmented, dim=0)
    # groundtruth_augmented = torch.cat(groundtruth_augmented, dim=0)
    # print(images_augmented.shape, groundtruth_augmented.shape)

    images_augmented = images_augmented / 255
    groundtruth_augmented = groundtruth_augmented / 255

    return images_augmented, groundtruth_augmented
   

class RoadSegmentationDataset(Dataset):
    def __init__(self, images_paths, groundtruth_paths, device='cpu'):
        self.images_list = []
        self.groundtruth_list = []

        for i, j in tqdm(zip(images_paths, groundtruth_paths)):
            img_i = load_image(i, cv.IMREAD_COLOR )
            img_j = load_image(j, cv.IMREAD_GRAYSCALE)
            if (img_j > 0).sum() > 1000:
                self.images_list.append(img_i)
                self.groundtruth_list.append(img_j)

        self.images = np.zeros([len(self.images_list), 400, 400, 3], dtype=np.float32)
        self.groundtruth = np.zeros([len(self.images_list), 400, 400], dtype=np.float32)

        for idx, (i, j) in tqdm(enumerate(zip(self.images_list, self.groundtruth_list))):
            self.images[idx] = i
            self.groundtruth[idx] = j

        self.images = (torch.tensor(self.images) / 255).permute(0, 3, 1, 2)
        self.groundtruth = (torch.tensor(self.groundtruth) > 0).to(torch.float)

        self.images = self.images.to(device)
        self.groundtruth = self.groundtruth.to(device)

    def __len__(self):
        assert len(self.images) == len(self.groundtruth)
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.groundtruth[index]

    def augment(self):
        self.images, self.groundtruth = aumgent_data(self.images, self.groundtruth)
        pass

if __name__ == '__main__':
    images_dir = 'training/images'
    groundtruth_dir = 'training/groundtruth'

    images_list = os.listdir(images_dir)[0:10]
    images_paths = [os.path.join(images_dir, i) for i in images_list]
    groundtruth_paths = [os.path.join(groundtruth_dir, i) for i in images_list]
    ds = RoadSegmentationDataset(images_paths, groundtruth_paths)
    ds.augment()
    print(ds[0])

    pass

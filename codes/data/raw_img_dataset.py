import os
import torch.nn as nn
import random
import torchvision

from PIL import Image
from torchvision.transforms import transforms


#Image resolution sizes based on training images
transform_low_res = transforms.Compose([
    transforms.Resize(int(80 * 1.14)),
    transforms.CenterCrop(80),
    transforms.ToTensor()
])

transform_high_res = transforms.Compose([
    transforms.Resize(int(320 * 1.14)),
    transforms.CenterCrop(320),
    transforms.ToTensor()
])

class Create_Dataset(nn.Module):
    def __init__(self, paths):
        self.img_paths = []
        cur_dir = os.getcwd()
        for path in paths:
            for filename in os.listdir(cur_dir + path):
                self.img_paths.append(cur_dir + path + "\\" + filename)
#        print(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, ind):
        image_raw = Image.open(self.img_paths[ind])

        low_res = transform_low_res(image_raw)
        high_res = transform_high_res(image_raw)
        return low_res, high_res    # format: image, label


class Image_Saver():
    def __init__(self, path):
        self.path = os.getcwd() + path
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def save_image(self, image_tensors):
        for tensor in image_tensors:
            image_path = self.path + "\\" + str(random.randint(0,200)) + ".png"
            torchvision.utils.save_image(tensor, image_path)

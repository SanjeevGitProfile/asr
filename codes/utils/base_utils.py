import os
import torch
import logging
import matplotlib.pyplot as plt

def get_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO)
    return logger

def get_curdir():
    return os.getcwd()

def plot_image(tensor):
    print(tensor[0].shape)
    plt.imshow(torch.squeeze(tensor[0])[0,:,:])
    plt.show()

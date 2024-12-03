import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

from utils import base_utils
from configs.configuration import Configuration
from data.raw_img_dataset import Create_Dataset, Image_Saver
from models.gan import SRGenerator, SRDiscriminator
from models.sr_loss import SuperResolutionLoss

logger = base_utils.get_logger('asr_train')
logger.info('Super resolution training')


def train():
    configs = Configuration()
    configs.show()

    # Prepare dataset
    dataset = Create_Dataset(configs.data_path)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print("Train size: ", train_size, " Test size: ", test_size)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size)

    # Shape of input tensors
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)
        break

    # Init Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    srgen_model = SRGenerator().to(device)
    optimizer = torch.optim.Adam(srgen_model.parameters(), weight_decay=1e-4)
    loss_estimator = torch.nn.MSELoss().to(device)

    # Custom loss function with Pixel and Perceptual loss
    # loss_estimator = SuperResolutionLoss(perceptual_weight=0.01).to(device)

    # Start training
    for epoch in range(configs.epochs):
        print("Epoch: ", epoch)
        srgen_model.train()

        for lr_img, hr_img in tqdm(train_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            optimizer.zero_grad()
            pred = srgen_model(lr_img)
            loss = loss_estimator(pred, hr_img)
            loss.backward()
            optimizer.step()
            #print("Training Loss: ", loss)

        # Evaluate on test data
        srgen_model.eval()

        for lr_img, hr_img in test_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            pred = srgen_model(lr_img)
            loss = loss_estimator(pred, hr_img)
            print("Test Loss: ", loss)

    # Save model
    torch.save(srgen_model.state_dict(),
               base_utils.get_curdir() + "\\models\\" + configs.model_name)

    # Predict & Save
    image_saver = Image_Saver("\\output\\itr" + str(configs.epochs))
    for lr_img, hr_img in test_loader:
        output = srgen_model(lr_img.to(device))
        print("output: ", output.shape)
        image_saver.save_image(output)
        break


if __name__ == '__main__':
    train()


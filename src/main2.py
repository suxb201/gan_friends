import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 方便复现
manual_seed = 999
print("Random Seed: ", manual_seed)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

BATCH_SIZE = 128
IMAGE_SIZE = 64
EPOCHS = 5
LR = 0.0002
BETA1 = 0.5  # Beta1 hyper param for Adam optimizers
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(64, 100, 1, 1, device=DEVICE)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # 100 -> 64x8
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 64 x 64
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1)
        return x


def train():
    generator = Generator().to(DEVICE)
    generator.apply(weights_init)
    discriminator = Discriminator().to(DEVICE)
    discriminator.apply(weights_init)
    criterion = nn.BCELoss()

    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))

    print("Starting Training Loop...")

    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):
            batch_size = data[0].size(0)

            real_label = torch.full((batch_size,), 1.0, dtype=torch.float, device=DEVICE)
            fake_label = torch.full((batch_size,), 0.0, dtype=torch.float, device=DEVICE)

            noise = torch.randn(batch_size, 100, 1, 1, device=DEVICE)
            fake_image = generator(noise)
            real_image = data[0].to(DEVICE)

            real_output = discriminator(real_image)
            fake_output = discriminator(fake_image.detach())

            loss_real = criterion(real_output, real_label)
            loss_fake = criterion(fake_output, fake_label)
            loss_d = loss_real + loss_fake
            discriminator.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            output = discriminator(fake_image)
            loss_g = criterion(output, real_label)
            generator.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch, EPOCHS, i, len(dataloader), loss_d.item(), loss_g.item()))
        # save
        with torch.no_grad():
            fake_image = generator(fixed_noise).detach().cpu()
            fake_image = torchvision.utils.make_grid(fake_image, padding=2, normalize=True)
            plt.imshow(fake_image)
            plt.axis('off')
        plt.savefig(f'images/image_at_epoch_{epoch:04}.png')


if __name__ == '__main__':
    dataset = torchvision.datasets.ImageFolder(root="../dataset/anime", transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Plot some training images
    real_batch = next(iter(dataloader))
    print(real_batch.shape)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            torchvision.utils.make_grid(real_batch[0][:64], padding=2, normalize=True),
            (1, 2, 0)
        ))

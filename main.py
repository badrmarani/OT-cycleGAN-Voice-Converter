import torch

import config
from src.dataset import dataset_loader
from src.model import Discriminator, Generator
from utils import train

loss_history = []
acc_history = []

GeneratorX = Generator().to(config.device)
GeneratorY = Generator().to(config.device)
DiscriminatorX = Discriminator().to(config.device)
DiscriminatorY = Discriminator().to(config.device)
opt_generator = torch.optim.Adam(
    list(GeneratorX.parameters()) + list(GeneratorY.parameters()),
    lr=config.lr,
    betas=(0.5, 0.999),
)
opt_discriminator = torch.optim.Adam(
    list(DiscriminatorX.parameters()) + list(DiscriminatorY.parameters()),
    lr=config.lr,
    betas=(0.5, 0.999),
)

train_loader = dataset_loader(config.batch_size, type="train")

# g_scaler = torch.cuda.amp.grad_scaler.GradScaler()
# d_scaler = torch.cuda.amp.grad_scaler.GradScaler()

for epoch in range(config.epochs):
    train(
        GeneratorX,
        GeneratorY,
        DiscriminatorX,
        DiscriminatorY,
        opt_generator,
        opt_discriminator,
        epoch,
        train_loader,
        config,
    )

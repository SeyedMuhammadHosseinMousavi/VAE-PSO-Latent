# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:55:47 2024

@author: seyed.mousavi
"""
%reset -f
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pyswarms.single.global_best import GlobalBestPSO
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# Set cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------
# Image folder path
image_folder_path = 'Depth'
# ---------------------------------------------------------------

# Image loader using torchvision's datasets.ImageFolder
image_size = 64
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root=image_folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class VAE(nn.Module):
    def __init__(self, image_channels=3, image_size=64, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        
        conv_output_size = image_size // 16
        self.conv_output_dim = 128 * conv_output_size * conv_output_size

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # input dimension for the first fully connected layer
        self.fc1 = nn.Linear(self.conv_output_dim, z_dim)  # Mean μ layer
        self.fc2 = nn.Linear(self.conv_output_dim, z_dim)  # Log variance σ layer
        # Decoder input should match the output z dimension
        self.fc3 = nn.Linear(z_dim, self.conv_output_dim)  # Prepares z for the decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    def decode(self, z):
        z = self.fc3(z)
        z = z.view(-1, 128, image_size // 16, image_size // 16)  # Reshape z to the input shape for the decoder
        return self.decoder(z)
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar

# Model, optimizer, and loss function
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training the VAE
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        recon_loss = nn.functional.mse_loss(recon_batch, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset):.4f}')

# Initiate the training process
num_epochs = 100  
for epoch in range(1, num_epochs + 1):
    train(epoch)

# Save a trained model (optional)
torch.save(model.state_dict(), 'vae.pth')

### PSO Optimization for Latent Vector
def reconstruct_images(target_imgs, model, iterations=200, n_particles=100, dimensions=32, save_path='reconstructed_images'):
    os.makedirs(save_path, exist_ok=True)
    # target_imgs is a batch of images [N, C, H, W]
    target_imgs_tensor = target_imgs.to(device)
    # Setup and run the PSO for each image separately
    for i in range(target_imgs_tensor.shape[0]):  # Loop through each image in the batch
        single_img_tensor = target_imgs_tensor[i].unsqueeze(0).repeat(n_particles, 1, 1, 1)
        # Define the objective function for PSO
        def f(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
            reconstructed_img = model.decode(x_tensor)  # Ensure this is (N, C, H, W)
            loss = torch.nn.functional.mse_loss(reconstructed_img, single_img_tensor, reduction='none').mean([1, 2, 3])
            return loss.cpu().detach().numpy()
        # Setup and run the PSO
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options)
        best_cost, best_pos = optimizer.optimize(f, iters=iterations)
        # Reconstruct the image using the optimized latent vector
        best_pos_tensor = torch.tensor(best_pos, dtype=torch.float32).unsqueeze(0).to(device)
        reconstructed_img = model.decode(best_pos_tensor).squeeze(0)
        # Save each reconstructed image
        img_save_path = os.path.join(save_path, f'reconstructed_{i}.png')
        save_image(reconstructed_img.clamp(0, 1), img_save_path)
        print(f"Image saved to {img_save_path}")

# Prepare a batch of target images
target_imgs = []  
for i in range(14):  #  to reconstruct 
    img_path = os.path.join(image_folder_path, f'image_{i}.png')  # Adjust as needed
    img = Image.open(img_path)
    img = transform(img)  # Apply the same transforms as before
    target_imgs.append(img)

target_imgs_tensor = torch.stack(target_imgs)  # Create a single tensor from the list
# Reconstruct the images using PSO-optimized latent vectors
model.eval()  # Set the model to evaluation mode
reconstruct_images(target_imgs_tensor, model, save_path='path_to_save_images2')

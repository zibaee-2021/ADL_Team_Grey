# GROUP19_COMP0197

import sys
import os
import json

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from utils.paths import *
from utils.device import get_optimal_device
from utils.model_init import initialise_weights
from mvae.data_handler import (
    Animals10Dataset,
    Reduced_ImageNetDataset,
    PatchMasker, 
    compute_loss
)
from shared_network_architectures.networks_pt import (
    get_network,
    SegmentModel
)

def process_batch(input_images, model, patch_masker, params, device):
    norm_pix_loss = True
    
    input_images = input_images.to(device)
    
    # Add random masking to the input images
    masked_images, masks = patch_masker.mask_patches(input_images)
    
    # Forward pass through the Masked Autoencoder
    logits = model(masked_images)
    
    # Apply sigmoid activation to get reconstructed pixel values
    reconstructed_images = torch.sigmoid(logits)
    
    # Normalize the input images (optional)
    if norm_pix_loss:
        mean = input_images.mean(dim=[-1, -2, -3], keepdim=True)
        var = input_images.var(dim=[-1, -2, -3], keepdim=True)
        input_images = (input_images - mean) / (var + 1.e-6) ** 0.5
    
    # Calculate the mean squared error loss per pixel
    loss = (reconstructed_images - input_images) ** 2
    loss = loss.mean(dim=[-1, -2, -3])  # [N], mean loss per pixel
    
    # Calculate the mean loss on removed patches
    mask_ratio = masks.sum() / masks.numel()
    loss = loss.sum() / (1.0 - mask_ratio)
    
    return loss

torch.manual_seed(42)
device = get_optimal_device()

with open('train_mvae_vit.json', 'r') as f:
    params = json.load(f)

experiment_name = f"{params["network"]}_{params["dataset_name"]}_pretrain_decoderv2"
wandb.init(name = f"{experiment_name}", project="last_Day", entity="adl_team_grey", config=params)

mask_ratio = params['mask_ratio']
pt_batch_size = params['pt_batch_size']
pt_num_epochs = params['pt_num_epochs']
pt_learning_rate = params['learning_rate']
pt_momentum = params['pt_momentum']
pt_optimizer = params['optimizer']

if params['image_size'] % params['patch_size'] != 0:
    raise ValueError(f"Patch size ({params['patch_size']}) does not divide image size ({params['image_size']})")

num_patches = (params['image_size'] // params['patch_size']) ** 2
num_masks = int(num_patches * mask_ratio)

# Data
data_dir = os.path.join(datasets_dir)
dataset_name = params['dataset_name']
if dataset_name == 'Animals10':
    data_dir = os.path.join(data_dir, 'Animals10/raw-img')
    pt_dataset = Animals10Dataset(data_dir)
else:
    data_dir = os.path.join(data_dir, 'imagenet-1k-resized')
    pt_dataset = Reduced_ImageNetDataset(data_dir)

# Split the training dataset into train_set, val_set
train_split_size = int(len(pt_dataset) // (100 / 95))
valid_split_size = int(len(pt_dataset) - train_split_size)

train_set, val_set = torch.utils.data.random_split(pt_dataset, [train_split_size, valid_split_size])
train_loader = DataLoader(train_set,
                            batch_size=params['pt_batch_size'],
                            shuffle=True,
                            drop_last=True,  # drop last batch so that all batches are complete
                            num_workers=1)
val_loader = DataLoader(val_set,
                            batch_size=params['pt_batch_size'],
                            shuffle=True,  # shuffle so it can demonstrate different images
                            drop_last=True,  # drop last batch so that all batches are complete
                            num_workers=1)

# Model
encoder, decoder = get_network(params, params['num_channels'])
initialise_weights(encoder)
initialise_weights(decoder)
vae_model = SegmentModel(encoder, decoder).to(device)  # vae pre-trainer and supervised fine-tuner share encoder
patch_masker = PatchMasker(params['patch_size'], num_masks)

optimizer = optim.Adam(vae_model.parameters(), lr=params['learning_rate'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Define early stopping parameters
best_val_loss = float('inf')
patience = 10
counter = 0

# Training
for epoch in range(pt_num_epochs):
    vae_model.train()
    train_loss = 0.0
    for its, input_images in enumerate(train_loader):
        loss = process_batch(input_images, vae_model, patch_masker, params, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        wandb.log({"Running Loss": loss.item()})

    train_loss /= len(train_loader)
    wandb.log({"Epoch": epoch + 1, "Training Loss": train_loss})

    # Validation
    vae_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for its, input_images in enumerate(val_loader):
            loss = process_batch(input_images, vae_model, patch_masker, params, device)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    wandb.log({"Epoch": epoch + 1, "Validation Loss": val_loss})

    if epoch % 10 == 0:
        test_images = patch_masker.rich_test(vae_model, val_loader, device)
        wandb.log({f"Example after {epoch} epochs": test_images})

    # Early stopping
    if val_loss < best_val_loss and epoch > 30: # add condition for if pretrained (will want to do earlier)
        print(f'Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(encoder.state_dict(), f'{experiment_name}_best_model_encoder.pt')
        torch.save(decoder.state_dict(), f'{experiment_name}_best_model_decoder.pt')
        best_val_loss = val_loss
    elif val_loss < best_val_loss:
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

vae_model.eval()
test_images = patch_masker.rich_test(vae_model, val_loader, device)
wandb.log({"After Training Example": test_images})
    
print("MVAE Script complete")


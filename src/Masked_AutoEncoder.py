"""
Implements Masked Auto Encoder using Vision Transformer: https://arxiv.org/abs/2111.06377
Resources:
 https://github.com/google-research/vision_transformer
 https://github.com/IcarusWizard/MAE/tree/main
"""

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import time
import os

import numpy as np
import matplotlib.pyplot as plt


network_parameters = { # just a placeholder
    "image_size": 256,
    "patch_size": 4, #must be divisor of image_size
    "num_classes": 1000,
    "num_layers": 6,
    "num_heads": 4,
    "hidden_dim": 256,
    "mlp_dim": 1024
}


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"





class Animals10Dataset(Dataset):
    def __init__(self, root_dir, target_size=(224, 224), transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            # Check if the class directory is a directory
            if not os.path.isdir(cls_dir):
                continue
            for filename in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, filename)
                images.append((img_path, self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path)

        # Ensure image has 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(self.target_size)
        # img = transforms.ToTensor()(img)

        img = img.resize(self.target_size)  # Resize the image
        if self.transform:
            img = self.transform(img)
        return img, label




class MaskedAutoEncoder_Encoder(nn.Module):
    def __init__(self, config, mask_ratio):
        super(MaskedAutoEncoder_Encoder, self).__init__()
        self.encoder = models.vision_transformer.VisionTransformer(image_size =config['image_size'],
                                                               patch_size=config['patch_size'],
                                                               num_classes=config['num_classes'],
                                                               num_layers=config['num_layers'],
                                                               num_heads=config['num_heads'],
                                                               hidden_dim=config['hidden_dim'],
                                                               mlp_dim=config['mlp_dim'])
        self.patch = Patch(config, mask_ratio)
    
    def forward(self, img, mask_ratio):
        patched = self.patch(img)
        encoded = self.encoder(patched)
        return encoded
    


class MaskedAutoEncoder_Decoder(nn.Module):
    def __init__(self, config):
        super(MaskedAutoEncoder_Decoder, self).__init__()
        self.decoder = nn.Linear(config['num_classes'], config['patch_size']*config['patch_size']*3)
        self.num_patches=(config['image_size'] // config['patch_size'])**2
        self.patch_size=config['patch_size']

    def forward(self, encoding):
        reconstructed_patches = self.decoder(encoding)
        reconstructed_patches = reconstructed_patches.view(-1, self.num_patches, 3, self.patch_size, self.patch_size)
        return reconstructed_patches
    


class Patch(nn.Module): # sets patches to black
    def __init__(self, config, mask_ratio):
        super().__init__()
        self.patch_size = config['patch_size']
        self.image_size = config['image_size']
        self.num_patches_per_side = (self.image_size // self.patch_size)
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_numbers = list(range(self.num_patches))
        self.mask_ratio = mask_ratio

    def forward(self, img):
        selected_patch_numbers = torch.randint(0, self.num_patches, (int(mask_ratio * self.num_patches),), dtype=torch.long) 
        for patch_number in selected_patch_numbers:
            # Calculate the coordinates of the top-left corner of the patch
            row = patch_number // self.num_patches_per_side
            col = patch_number % self.num_patches_per_side
            top_left_x = col * self.patch_size
            top_left_y = row * self.patch_size
            img[top_left_y:top_left_y+self.patch_size, top_left_x:top_left_x+self.patch_size, :] = 0
        return img


class MaskedAutoEncoder_VisionTransformer(nn.Module):
    def __init__(self, config, mask_ratio):
        super(MaskedAutoEncoder_VisionTransformer, self).__init__()
        self.encoder = MaskedAutoEncoder_Encoder(config, mask_ratio)
        self.decoder = MaskedAutoEncoder_Decoder(config)

    def forward(self, input): # returns a decoded encoding of the original image
        features = self.encoder(input, mask_ratio)
        output = self.decoder(features)
        return output


if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.abspath(__file__))
    device = get_optimal_device()

    # Some definitions
    data_dir = script_directory+"/Animals-10/raw-img/"
    image_size=256
    mask_ratio = 0.75
    batch_size=50
    num_epochs=1
    report_every=100
    learning_rate = 0.01
    momentum = 0.9


    # Define transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    # load Animals-10 dataset
    dataset = Animals10Dataset(data_dir, (network_parameters['image_size'], network_parameters['image_size']), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)





    #criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # consider Adam?

    # Create data loaders and classes
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get a random batch from the DataLoader
    images, labels = next(iter(train_loader))
    # Select a random image from the batch
    random_index = np.random.randint(len(images))
    random_image = images[random_index]

    # Convert the image tensor to numpy array and transpose it to (H, W, C) format
    random_image_np = random_image.permute(1, 2, 0).numpy()

    # Display the selected image
    plt.axis('off')
    plt.imshow(random_image_np)
    plt.show()

    patch_images = Patch(network_parameters, mask_ratio)
    patched_img = patch_images(random_image_np)
    plt.axis('off')
    plt.imshow(patched_img)
    plt.show()

    # Instantiate the Masked Autoencoder model
    model = MaskedAutoEncoder_VisionTransformer(network_parameters, mask_ratio)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    start_time=time.time()
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, _ = data  # Ignore labels for unsupervised training
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # outputs = mae_model(inputs)
            loss = criterion(outputs, inputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % report_every == (report_every-1):  # Print every report_every mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / report_every:.4f}")
                running_loss = 0.0
        end_time=time.time()
        print(f"Epoch {epoch + 1}. Running for {(end_time-start_time):.0f}s so far")

    # Put something to save the model
    
    # Next stage, feed onto classifier
        
    print(f"Finished training after {(end_time-start_time):.0f}s")




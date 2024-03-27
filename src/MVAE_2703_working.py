import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from PIL import Image
import os
import time

import random
import numpy as np
import matplotlib.pyplot as plt

"""
TO DO
1 Review position embedding (ViT)
2 Review decoder architecture
3 build in config pictionary
4 Test model with inference and in-filling
5 build in optimal device
6 review which libraries are necessary
7 move .view into model class
"""


network_parameters = {      # just a placeholder structure
    "image_size": 224,      # number of pixels square
    "num_channels": 3,      # RGB image -> 3 channels
    "patch_size": 16,       # must be divisor of image_size
    "num_features": 1000,   # number of features created by the vision transformer
    "num_layers": 6,        # ViT parameter
    "num_heads": 4,         # ViT parameter
    "hidden_dim": 256,      # ViT parameter
    "mlp_dim": 1024         # ViT parameter
}


def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def mask_tester(patch_masker, image_file):
    image = Image.open(image_file)
    image = image.resize((224,224))
    transform = transforms.ToTensor()
    input_image = transform(image).unsqueeze(0)
    masked_image_tensor = patch_masker.mask_patches(input_image)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].axis('off')
    axes[1].axis('off')
    axes[0].imshow(input_image[0].numpy().transpose((1,2,0)))
    axes[1].imshow(masked_image_tensor[0].numpy().transpose((1,2,0)))
    plt.show()



"""
This class loads in the images and cleans it up a bit
"""
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



"""
This class defines the encoder / decoder
"""
# Define the masked autoencoder model
class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define your Vision Transformer-based encoder for color images
class VisionTransformerEncoder(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes):
        super(VisionTransformerEncoder, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, num_classes, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_classes, image_size // patch_size, image_size // patch_size))

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        return x

class NewVisionTransformerEncoder(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes):
        super(NewVisionTransformerEncoder, self).__init__()
        self.vit = models.vision_transformer.VisionTransformer(image_size = image_size,   # Load the pretrained ViT model
                                                               patch_size=patch_size,
                                                               num_classes=num_classes,
                                                               num_layers=4,
                                                               num_heads=4,
                                                               hidden_dim=256,
                                                               mlp_dim=512)

    def forward(self, x):
        # Pass the input through the ViT-B_16 backbone
        features = self.vit(x)
        return features


# Define your decoder for color images
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_one, hidden_dim_two):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_one)
        self.fc2 = nn.Linear(hidden_dim_one, hidden_dim_two)
        self.fc3 = nn.Linear(hidden_dim_two, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.relu(self.fc1(x)) ## sort out dimensions
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x



class PatchMasker:
    def __init__(self, patch_size=32, num_patches_to_mask=5):
        self.patch_size = patch_size
        self.num_patches_to_mask = num_patches_to_mask

    def mask_patches(self, image):
        """
        Divide the image into non-overlapping patches and randomly mask a fixed number of patches.

        Args:
            image (torch.Tensor): The input color image represented as a PyTorch tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The masked image.
        """
        batch_size, channels, height, width = image.size()

        # Determine the number of patches in height and width
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Create a list of all patch indices
        patch_indices = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]

        # Randomly choose non-repeating patch indices to mask
        masked_patch_indices = random.sample(patch_indices, self.num_patches_to_mask)

        # Apply the mask to the image
        masked_image = image.clone()
        for index in masked_patch_indices:
            i, j = index
            y_start = i * self.patch_size
            y_end = min((i + 1) * self.patch_size, height)
            x_start = j * self.patch_size
            x_end = min((j + 1) * self.patch_size, width)
            masked_image[:, :, y_start:y_end, x_start:x_end] = 0  # Set masked patch to zero

        return masked_image


if __name__ == '__main__':

    
    # Definitions
    ##############
    torch.manual_seed(42) # Set seed for random number generator
    # image and masking
    image_size = 224
    num_channels = 3
    mask_ratio = 0.75
    patch_size = 1
    if image_size % patch_size !=0: print("Alert! Patch size does not divide image size")
    num_patches = (image_size // patch_size) **2
    num_masks = int(num_patches * mask_ratio)
    # training
    batch_size=10
    report_every=100
    num_epochs=2
    report_every=100
    learning_rate = 0.01
    momentum = 0.9

    # Instantiate the encoder for color images
    encoder = NewVisionTransformerEncoder(image_size=224, patch_size=16, in_channels=3, num_classes=768)  # 3 channels for RGB images, num_classes should be chosen as per ViT configuration
    # Instantiate the decoder
    decoder = Decoder(input_dim=768, output_dim=224*224*3, hidden_dim_one=512, hidden_dim_two=1024)  # Adjust input_dim as per your ViT configuration
    # Instantiate PatchMasker
    patch_masker = PatchMasker(patch_size, num_masks)
    # Create the masked autoencoder model
    model = MaskedAutoencoder(encoder, decoder)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load and preprocess the dataset
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_directory+"/Animals-10/raw-img/"


    # Define your dataset and dataloader
    # Define transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    # load Animals-10 dataset
    dataset = Animals10Dataset(data_dir, (image_size, image_size), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) # drop last batch so that all batches are complete



    # mask_tester(patch_masker, "/users/Stephen/My Articles/Machine Learning and Artificial Intelligence/UCL ML MSc/Assignments/ADL - Group/Code/Animals-10/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")


    # Training loop
    start_time=time.time()
    losses=[]
    for epoch in range(num_epochs):
        epoch_start_time=time.time()
        running_loss = 0.0
        for its, (input_images, input_labels) in enumerate(dataloader):
            # Add random masking to the input images
            masked_images = patch_masker.mask_patches(input_images)

            # Forward pass
            output_tensors = model(masked_images)
            outputs = output_tensors.view(batch_size, num_channels, image_size, image_size) # put this into class

            # Compute the loss
            loss = criterion(outputs, input_images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if its % report_every == (report_every-1):    # print every report_every mini-batches
                print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], running loss: %.3f' % (epoch + 1, num_epochs, batch_size, its + 1, len(dataloader) ,running_loss / len(dataloader)))
        epoch_end_time=time.time()
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {(epoch_end_time-epoch_start_time):.0f}s, Loss: {running_loss / len(dataloader):.3f}")
        losses.append(running_loss / len(dataloader))
    print(f"Training finished after {(epoch_end_time-start_time):.0f}s")

    # Save the trained model
    model_file = "masked_autoencoder.pt"
    if os.path.exists(model_file):
        os.remove(model_file)
    torch.save(model.state_dict(), model_file)
    print("File saved\nFinished")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
from PIL import Image
import os
import time
import json

import random
import numpy as np
import matplotlib.pyplot as plt

"""
TO DO
1 Review position embedding (ViT) parameters
2 Review decoder architecture
3 IMPORTANT - Segmenter network (convolution layer?)
4 move .view into model class
5 load saved model before continuing training
6 Segmentation inference & visualisation
7 speedup? 
8 review necessary libraries
"""


def get_optimal_device():
    """
    Returns: optimal device.
        CUDA (GPU) if available, MPS if mac, CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mask_tester(patch_masker, image_file):
    """
    Takes
        patch masker object, link to image file
    Displays
        original and patched image
    Returns
        input image, masked image tensor
    """
    image = Image.open(image_file)
    image = image.resize((224, 224))
    transform = transforms.ToTensor()
    input_image = transform(image).unsqueeze(0)
    masked_image_tensor = patch_masker.mask_patches(input_image)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].axis('off')
    axes[1].axis('off')
    axes[0].imshow(input_image[0].numpy().transpose((1, 2, 0)))
    axes[1].imshow(masked_image_tensor[0].numpy().transpose((1, 2, 0)))
    plt.show()

    return input_image[0], masked_image_tensor


def segmentation_tester():
    original_image_tensor, masked_image_tensor = mask_tester(patch_masker,
                                                             data_dir + "/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
    original_image_tensor = original_image_tensor.unsqueeze(0)
    segmentation_map = segmentation_model(original_image_tensor)
    # Get the predicted segmentation mask
    segmentation_mask = segmentation_map.argmax(dim=1).squeeze()
    # Define color map for different classes
    # Dictionary mapping class labels to RGB colors
    colors = {0: [255, 0, 0],  # Class 0: Red
              1: [0, 255, 0],  # Class 1: Green
              2: [0, 0, 255]  # Class 2: Blue
              }  # Add more class labels and corresponding colors as needed
    # Create an empty image with the same size as the original image
    overlay = torch.zeros(3, 224, 224)
    # Overlay each class with a different color
    for class_label, color in colors.items():
        # Mask for pixels corresponding to the current class label
        mask = segmentation_mask == class_label
        overlay[:, mask] = torch.tensor(color, dtype=overlay.dtype).unsqueeze(1)

    # Convert PyTorch tensors to NumPy arrays
    image1_np = original_image_tensor[0].permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array (HWC format)
    image2_np = overlay.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array (HWC format)

    # Scale pixel values from [0, 1] to [0, 255] and convert to uint8
    image1_np_uint8 = (image1_np * 255).astype(np.uint8)
    image2_np_uint8 = (image2_np * 255).astype(np.uint8)

    # Convert NumPy arrays to Pillow images
    image1_pil = Image.fromarray(image1_np_uint8)
    image2_pil = Image.fromarray(image2_np_uint8)

    # Blend the two Pillow images
    highlighted_image = Image.blend(image1_pil, image2_pil, alpha=0.7)

    # Combine the original image with the overlay
    # highlighted_image = Image.blend(transforms.ToPILImage(original_image_tensor), transforms.ToPILImage(overlay), alpha=0.5)
    # Visualize the highlighted image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(original_image_tensor[0].permute(1, 2, 0))  # Assuming image_tensor is in CHW format
    plt.title('Original Image')
    #
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(image2_np)
    plt.title('Overlay')
    #
    plt.subplot(1, 3, 3)
    plt.imshow(highlighted_image)
    plt.axis('off')
    plt.title('Blend')
    plt.show()


class Animals10Dataset(Dataset):
    """
    Loads and cleans up images in Animals-10 dataset
    """

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


class MaskedAutoencoder(nn.Module):
    """
    Defines encoder / decoder for masked autoencoder pre-trainer
    Takes
        batch of images
    Returns
        image passed through encoder and decoder
    """

    def __init__(self, encoder, decoder):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VisionTransformerEncoder(nn.Module):
    """
    Defines vision transformer model
    Takes
        batch of images
    Returns
        vision-transformer feature embeddings of those images
    """

    def __init__(self, parameters):
        super(VisionTransformerEncoder, self).__init__()
        self.image_size = parameters['image_size']
        self.patch_size = parameters['patch_size']
        self.num_features = parameters['vit_num_features']
        self.num_layers = parameters['vit_num_layers']
        self.num_heads = parameters['vit_num_heads']
        self.hidden_dim = parameters['vit_hidden_dim']
        self.mlp_dim = parameters['vit_mlp_dim']
        self.vit = models.vision_transformer.VisionTransformer(image_size=self.image_size,
                                                               patch_size=self.patch_size,
                                                               num_classes=self.num_features,
                                                               num_layers=self.num_layers,
                                                               num_heads=self.num_heads,
                                                               hidden_dim=self.hidden_dim,
                                                               mlp_dim=self.mlp_dim)

    def forward(self, x):
        # Pass the input through the ViT-B_16 backbone
        features = self.vit(x)
        return features


# Define color image decoder
class VisionTransformerDecoder(nn.Module):
    """
    Decoder
    Takes
       batch of image feature embeddings
    Returns
        reconstructed image tensor
    """

    def __init__(self, parameters):
        super(VisionTransformerDecoder, self).__init__()
        self.input_dim = parameters['vit_num_features']
        self.output_dim = parameters['image_size'] * parameters['image_size'] * parameters['num_channels']
        self.hidden_dim_one = parameters['decoder_hidden_dim_1']
        self.hidden_dim_two = parameters['decoder_hidden_dim_2']
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_one)
        self.fc2 = nn.Linear(self.hidden_dim_one, self.hidden_dim_two)
        self.fc3 = nn.Linear(self.hidden_dim_two, self.output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class PatchMasker:
    """
    Divide the image into non-overlapping patches and randomly mask a fixed number of patches.
    Args:
        image (torch.Tensor): The input color image represented as a PyTorch tensor with shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: The masked image.
    """

    def __init__(self, patch_size=32, num_patches_to_mask=5):
        self.patch_size = patch_size
        self.num_patches_to_mask = num_patches_to_mask

    def mask_patches(self, image):
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


def OxfordDataset(data_dir, train_split, val_split, test_split, batch_size):
    """
    Reads in Oxford IIIT-pet images and splits in training, validation, test data loaders
    """
    # Load the dataset using ImageFolder
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Calculate the sizes for train, validation, and test sets
    num_samples = len(dataset)
    num_train = int(train_split * num_samples)
    num_val = int(val_split * num_samples)
    num_test = num_samples - num_train - num_val
    # Split the dataset into train, validation, and test sets
    train_data, val_data, test_data = random_split(dataset, [num_train, num_val, num_test])
    # Create data loaders for each set
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    return train_loader, val_loader, test_loader


class SemanticSegmenter(nn.Module):
    """
    Takes
        batch of images
    Returns
        class probability per channel-pixel
    """

    def __init__(self, encoder, decoder):
        super(SemanticSegmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SegmentationDecoder(nn.Module):
    """
    Takes
        ferature embedding
    Returns
        class probability per channel-pixel
    """

    def __init__(self, parameters, num_classes):
        super(SegmentationDecoder, self).__init__()
        self.image_size = parameters['image_size']
        self.in_features = parameters['vit_num_features']
        self.hidden_dim_one = parameters['segmenter_hidden_dim_1']
        self.hidden_dim_two = parameters['segmenter_hidden_dim_2']
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.in_features, self.hidden_dim_one)
        #  take this out for now self.fc2 = nn.Linear(self.hidden_dim_one, self.hidden_dim_two)
        self.fc3 = nn.Linear(self.hidden_dim_two, num_classes * self.image_size * self.image_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # take this out for now x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = x.reshape(-1, self.num_classes, self.image_size, self.image_size)

        return x


#########
# Control
run_pretraining = False
check_masking = False
check_infilling = False
run_semantic_training = True
check_semantic_segmentation = False

if __name__ == '__main__':

    torch.manual_seed(42)  # Set seed for random number generator
    device = get_optimal_device()
    #############
    # Definitions
    parameters = {
        # image
        "image_size": 224,  # number of pixels square
        "num_channels": 3,  #  RGB image -> 3 channels
        "patch_size": 16,  # must be divisor of image_size
        # vision transformer encoder
        "vit_num_features": 768,  # number of features created by the vision transformer
        "vit_num_layers": 4,  # ViT parameter
        "vit_num_heads": 4,  # ViT parameter
        "vit_hidden_dim": 256,  # ViT parameter
        "vit_mlp_dim": 512,  # ViT parameter
        # vision transformer decoder
        "decoder_hidden_dim_1": 512,  # ViT decoder first hidden layer dimension
        "decoder_hidden_dim_2": 1024,  # ViT decoder second hidden layer dimension
        # segmentation model
        "segmenter_hidden_dim_1": 1024,  # segmentation model - more work needed on architecture - convolution?
        "segmenter_hidden_dim_2": 1024,
    }

    # image and masking
    mask_ratio = 0.75
    if parameters['image_size'] % parameters['patch_size'] != 0: print("Alert! Patch size does not divide image size")
    num_patches = (parameters['image_size'] // parameters['patch_size']) ** 2
    num_masks = int(num_patches * mask_ratio)
    # training
    pt_batch_size = 8
    report_every = 100
    pt_num_epochs = 20
    pt_learning_rate = 0.001
    pt_momentum = 0.9  # not used, since using Adam optimizer
    # file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir + "/Animals-10/raw-img/"
    model_file = "/masked_autoencoder_model.pth"
    encoder_file = "/masked_autoencoder_encoder.pth"
    decoder_file = "/masked_autoencoder_decoder.pth"
    segmentation_model_file = "/semantic_segmentation_model.pth"
    segmentation_decoder_file = "/segmentation_decoder.pth"
    oxford_dir = "/Oxford-IIIT-Pet_split/images"
    animals_classes = ('cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno',
                       'scoiattolo')  #  not used, since pre-training is (unsupervised) variational auto-encoder
    oxford_classes = (
    'Abyssinian_cat', 'american_bulldog_dog', 'american_pit_bull_terrier_dog', 'basset_hound_dog', 'beagle_dog',
    'Bengal_cat', 'Birman_cat', 'Bombay_cat', 'boxer_dog', 'British_Shorthair_cat', 'chihuahua_dog', 'Egyptian_Mau_cat',
    'english_cocker_spaniel_dog', 'english_setter_dog', 'german_shorthaired_dog', 'great_pyrenees_dog', 'havanese_dog',
    'japanese_chin_dog', 'keeshond_dog', 'leonberger_dog', 'Maine_Coon_cat', 'miniature_pinscher_dog',
    'newfoundland_dog',
    'Persian_cat', 'pomeranian_dog', 'pug_dog', 'Ragdoll_cat', 'Russian_Blue_cat', 'saint_bernard_dog', 'samoyed_dog',
    'scottish_terrier_dog', 'shiba_inu_dog', 'Siamese_cat', 'Sphynx_cat', 'staffordshire_bull_terrier_dog',
    'wheaten_terrier_dog', 'yorkshire_terrier_dog')
    ##############
    # fine tuning
    # datasets
    train_size = 0.8
    val_size = 0.1
    test_size = 1.0 - train_size - val_size
    ft_num_classes = len(oxford_classes)
    # training
    ft_batch_size = 8
    ft_num_epochs = 5
    ft_lr = 0.001

    # Instantiate the encoder & decoder for color images, patchmaker and model (encoder/decoder)
    encoder = VisionTransformerEncoder(parameters)  # num_classes == number of features in ViT image embedding
    decoder = VisionTransformerDecoder(parameters)
    patch_masker = PatchMasker(parameters['patch_size'], num_masks)
    pt_model = MaskedAutoencoder(encoder, decoder).to(device)

    # Define loss function and optimizer
    pt_criterion = nn.MSELoss()
    pt_optimizer = torch.optim.Adam(pt_model.parameters(), lr=pt_learning_rate)

    # load and pre-process Animals-10 dataset and dataloader & transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Animals10Dataset(data_dir, (parameters['image_size'], parameters['image_size']), transform=transform)
    dataloader = DataLoader(dataset, batch_size=pt_batch_size, shuffle=True,
                            drop_last=True)  # drop last batch so that all batches are complete

    #####################
    # Demonstrate masking
    if check_masking:
        original_image_tensor, masked_image_tensor = mask_tester(patch_masker,
                                                                 data_dir + "/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")

    ###############
    #  MVAE training
    if run_pretraining:  # Run training
        print("In pre-training")
        start_time = time.time()
        losses = []
        for epoch in range(pt_num_epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            for its, (input_images, input_labels) in enumerate(dataloader):
                input_images, input_labels = input_images.to(device), input_labels.to(device)
                # Add random masking to the input images
                masked_images = patch_masker.mask_patches(input_images)

                # Forward pass & compute the loss
                output_tensors = pt_model(masked_images)
                outputs = output_tensors.view(pt_batch_size, parameters['num_channels'], parameters['image_size'],
                                              parameters['image_size'])  # put this into class
                loss = pt_criterion(outputs, input_images)

                # Backward pass and optimization
                pt_optimizer.zero_grad()
                loss.backward()
                pt_optimizer.step()

                running_loss += loss.item()
                if its % report_every == (report_every - 1):  # print every report_every mini-batches
                    print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], running loss: %.4f' % (
                    epoch + 1, pt_num_epochs, pt_batch_size, its + 1, len(dataloader), running_loss / len(dataloader)))
            epoch_end_time = time.time()
            print(
                f"Epoch [{epoch + 1}/{pt_num_epochs}] completed in {(epoch_end_time - epoch_start_time):.0f}s, Loss: {running_loss / len(dataloader):.4f}")
            losses.append(running_loss / len(dataloader))
        print(f"Masked VAE training finished after {(epoch_end_time - start_time):.0f}s")

        # Save the trained model & losses
        torch.save(pt_model.state_dict(), script_dir + model_file)
        torch.save(encoder.state_dict(), script_dir + encoder_file)
        torch.save(decoder.state_dict(), script_dir + decoder_file)
        with open(script_dir + "pt_losses.txt", 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')
        print("Models saved\nFinished")


    else:  #  load pre-saved models
        print("Loading pre-saved models")
        encoder.load_state_dict(torch.load(script_dir + encoder_file), strict=False)
        decoder.load_state_dict(torch.load(script_dir + decoder_file), strict=False)
        pt_model.load_state_dict(torch.load(script_dir + model_file), strict=False)

    ################################
    # Demonstrate infilling an image
    if check_infilling:
        original_image_tensor, masked_image_tensor = mask_tester(patch_masker,
                                                                 data_dir + "/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
        with torch.no_grad():
            infill_image_tensor = pt_model(masked_image_tensor)
            infill_image_tensor = infill_image_tensor.reshape(parameters['num_channels'], parameters['image_size'],
                                                              parameters['image_size'])
            inpainted_image_tensor = masked_image_tensor[0] + infill_image_tensor * (
                        masked_image_tensor[0] == 0).float()

        # Visualize the results (assuming you have matplotlib installed)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 4, 1)
        plt.axis('off')
        plt.imshow(original_image_tensor.permute(1, 2, 0))  # Assuming image_tensor is in CHW format
        plt.title('Original Image')
        #
        plt.subplot(1, 4, 2)
        plt.axis('off')
        plt.imshow(masked_image_tensor[0].permute(1, 2, 0))
        plt.title('Masked image')
        #
        plt.subplot(1, 4, 3)
        plt.axis('off')
        plt.imshow(infill_image_tensor.permute(1, 2, 0))  # Assuming inpainted_image_tensor is in CHW format
        plt.title('Infill Image')
        #
        plt.subplot(1, 4, 4)
        plt.axis('off')
        plt.imshow(inpainted_image_tensor.permute(1, 2, 0))  # Assuming inpainted_image_tensor is in CHW format
        plt.title('Inpainted Image')
        #
        plt.show()
        plt.savefig(script_dir + '/infilling.png')
        #
        print("Infilling finished")

    # load data
    train_loader, val_loader, test_loader = OxfordDataset(script_dir + oxford_dir, train_size, val_size, test_size,
                                                          ft_batch_size)
    # Initialize model
    segmentation_decoder = SegmentationDecoder(parameters, ft_num_classes)
    ft_segmentation_model = SemanticSegmenter(encoder, segmentation_decoder).to(device)

    #############################
    # train semantic segmentation
    if run_semantic_training:
        print("In semantic segmentation training")  # images need to be in one folder per class
        start_time = time.time()
        ft_segmentation_model.train()

        # Define loss function and optimizer
        ft_criterion = nn.CrossEntropyLoss()
        ft_optimizer = torch.optim.Adam(ft_segmentation_model.parameters(), lr=ft_lr)

        losses = []
        for epoch in range(ft_num_epochs):
            epoch_start_time = time.time()
            running_loss = 0
            for its, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # forward pass
                outputs = ft_segmentation_model(images)
                outputs = outputs.reshape(ft_batch_size, ft_num_classes, -1)
                imagewide_probs = torch.mean(outputs, dim=2)
                # imagewide_probs = torch.mean(outputs, dim=(2, 3))
                loss = ft_criterion(imagewide_probs, labels)

                # backward pass
                ft_optimizer.zero_grad()
                loss.backward()
                ft_optimizer.step()

                running_loss += loss.item()
                if its % report_every == (report_every - 1):  # print every report_every mini-batches
                    print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], running loss: %.4f' % (
                    epoch + 1, ft_num_epochs, ft_batch_size, its + 1, len(train_loader),
                    running_loss / len(train_loader)))
            epoch_end_time = time.time()
            losses.append(running_loss / len(train_loader))
            print(
                f"Epoch [{epoch + 1}/{ft_num_epochs}] completed in {(epoch_end_time - epoch_start_time):.0f}s, Loss: {running_loss / len(train_loader):.4f}")
        print(f"Segmentation training finished after {(epoch_end_time - start_time):.0f}s")

        # save the trained model and losses
        torch.save(segmentation_decoder.state_dict(), script_dir + segmentation_decoder_file)
        torch.save(ft_segmentation_model.state_dict(), script_dir + segmentation_model_file)
        with open(script_dir + "ft_losses.txt", 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')
        print("Fine tune models saved\nFinished")


    else:  #  load pre-saved models
        print("Loading pre-saved models")
        segmentation_decoder.load_state_dict(torch.load(script_dir + segmentation_decoder_file), strict=False)
        encoder.load_state_dict(torch.load(script_dir + encoder_file), strict=False)
        ft_segmentation_model.load_state_dict(torch.load(script_dir + segmentation_model_file), strict=False)

    ###################################
    # demonstrate semantic segmentation
    if check_semantic_segmentation:
        print("Test semantic segmentation")

        ft_segmentation_model.eval()
        original_image_tensor, masked_image_tensor = mask_tester(patch_masker,
                                                                 data_dir + "/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
        with torch.no_grad():
            segmented_outputs = ft_segmentation_model(original_image_tensor)
        # Get the predicted segmentation mask
        segmentation_mask = segmented_outputs.argmax(dim=1).squeeze().numpy()

        # Define color map for different classes
        colors = [  # Add more colors for additional classes if needed
            (0, 0, 255),  # Class 0: Red
            (0, 255, 0),  # Class 1: Green
            (255, 0, 0),  # Class 2: Blue
        ]

        # Overlay each class with a different color
        overlay = np.zeros_like(original_image_tensor)

        for i, color in enumerate(colors):
            overlay[segmentation_mask == i] = color

        # Combine the original image with the overlay
        highlighted_image = Image.blend(Image.fromarray(original_image_tensor), Image.fromarray(overlay), alpha=0.5)

        # Visualize the highlighted image
        plt.imshow(highlighted_image)
        plt.axis('off')
        plt.show()

        print("Semantic segmentation finished")

    print("Script finished")
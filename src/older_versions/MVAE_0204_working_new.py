import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
from PIL import Image
import os
import time

import random
import numpy as np
# matplotlib.use("QtAgg") # problem with standard backend  https://stackoverflow.com/questions/42790212/matplotlib-doesnt-show-plots-on-mac-plt-show-hangs-on-macosx-backend
import matplotlib.pyplot as plt

"""
TO DO
1 Review position embedding (ViT) parameters
2 Review decoder architecture
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

    return input_image[0], masked_image_tensor



def view_batch(data_loader):

    for batch in data_loader:
        images, labels = batch
        print("Batch size:", images.size(0))

        # Convert one-hot labels to image
        labels_tensor=one_hot_to_tensor(labels)
        # Plot each image in the batch
        for i in range(images.size(0)):
            fig, axes = plt.subplots(1,2,figsize=(10,5))
            axes[0].imshow(images[i].permute(1,2,0))  # Matplotlib expects image in (H, W, C) format
            axes[1].imshow(labels_tensor[i].unsqueeze(0).permute(1,2,0))
            plt.tight_layout()
            plt.show()
        break  # Exit th


def view_training(model, test_loader):
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        outputs = model(images)

        #images = (images.cpu().detach().numpy()*255).astype(np.uint8)
        # labels = (labels.cpu().detach().numpy()*255).astype(np.uint8)
        output_labels = one_hot_to_tensor(outputs.cpu().detach())
        fig, axes = plt.subplots(3, 4, figsize=(12,8))
        for i in range(4):
            ax = axes[0,i]
            ax.axis('off')
            ax.imshow(images[i].cpu().permute(1,2,0))
            ax = axes[1,i]
            ax.axis('off')
            ax.imshow(labels[i][0])
            ax = axes[2,i]
            ax.axis('off')
            ax.imshow(output_labels[i].unsqueeze(0).permute(1,2,0))
        plt.tight_layout()
        if check_semantic_segmentation:
            plt.show()
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        plt.savefig(script_dir+'/labels'+date_str+'.png')
        break


def one_hot_to_tensor(one_hot):
    """
    Converts the one-hot segmented image produced by the model into a numpy array
    """
    if len(one_hot.shape) == 3: #no batch dimension
        tensor = torch.argmax(one_hot, dim=0).float()
        tensor /= one_hot.size(0)-1 # normalise to [0,1]
    elif len(one_hot.shape) == 4: # has batch dimension
        tensor = torch.argmax(one_hot, dim=1).float()
        tensor /= one_hot.size(1)-1
    else:
        raise ValueError ("Invalude input tensor shape")

    return tensor


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
    def __init__(self, params):
        super(VisionTransformerEncoder, self).__init__()
        self.image_size = params['image_size']
        self.patch_size = params['patch_size']
        self.num_features = params['vit_num_features']
        self.num_layers = params['vit_num_layers']
        self.num_heads = params['vit_num_heads']
        self.hidden_dim = params['vit_hidden_dim']
        self.mlp_dim = params['vit_mlp_dim']
        self.vit = models.vision_transformer.VisionTransformer(image_size = self.image_size,
                                                               patch_size = self.patch_size,
                                                               num_classes = self.num_features,
                                                               num_layers = self.num_layers,
                                                               num_heads = self.num_heads,
                                                               hidden_dim = self.hidden_dim,
                                                               mlp_dim = self.mlp_dim)

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
        reconstructed image tensor: Batch * Channels (3) * Image size (224) * Image size (224)

    In the MVAE process the three output dimensions are the three colours,
    in the segmentation process these get re-interpretted as probabilities
    over the three classes (foreground, background, boundary) - bit of a hack
    """
    def __init__(self, params):
        super(VisionTransformerDecoder, self).__init__()
        self.input_dim = params['vit_num_features']
        self.image_size = params['image_size']
        self.num_channels = params['num_channels']
        self.output_dim = self.image_size * self.image_size * self.num_channels
        self.hidden_dim_one = params['decoder_hidden_dim']
        self.CNN_channels = params['decoder_CNN_channels']
        self.upscale = params['decoder_scale_factor']
        self.CNN_patch = int(self.image_size / 2**2 / self.upscale)
        self.hidden_dim_two = int(self.CNN_channels * self.CNN_patch**2) # 5 upsampling layers, halving channels (except last layer)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_one)
        self.fc2 = nn.Linear(self.hidden_dim_one, self.CNN_channels * self.CNN_patch * self.CNN_patch)
        self.unflatten = nn.Unflatten(1, (self.CNN_channels,self.CNN_patch,self.CNN_patch))
        self.conv1 = nn.ConvTranspose2d(self.CNN_channels, int(self.CNN_channels/2), kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(int(self.CNN_channels/2), self.num_channels, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Unecessary??? Flatten the input tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.unflatten(x)
        x = self.relu(self.conv1(x))    # Output: B * 16 * 14 * 14
        x = self.upsample = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False) # Output: B * 8 * 112 * 112
        x = self.sigmoid(self.conv2(x)) # Output: B * 3 * 224 * 224
        x = x.view(-1, self.num_channels, self.image_size, self.image_size) # shouldnt be necessary
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



class OxfordPetDataset(Dataset):
    """
    Takes
        links to jpg images and trimap pngs
    Returns
        image tensors and one-hot classification map
    """
    def __init__(self, image_dir, label_dir, parameters):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.parameters = parameters
        self.image_size = parameters['image_size']
        self.segment_classes = parameters['segmenter_classes']
        self.image_filenames = [filename for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace('.jpg', '.png'))

        # read in image and convert to tensor
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),transforms.ToTensor(),])
        image_tensor = transform(image)

        # read in trimap, 1=foreground, 2=background, 3=indeterminate/boundary
        trimap = Image.open(label_path)
        label_transform = transforms.Compose([transforms.Resize((224, 224)), # Convert image to tensor # Scale pixel values to [0, 255] and convert to uint8
                                              transforms.ToTensor(),
                                              transforms.Lambda(lambda x: (x * 255).to(torch.uint8))])
        trimap_tensor = label_transform(trimap)

        # Create one-hot label encoding, including background and indeterminate
        segment_labels = torch.zeros((self.segment_classes,) + trimap_tensor.shape[1:], dtype=torch.int)
        segment_labels[0, :] = (trimap_tensor[0] == 1)  #foregrount
        segment_labels[1, :] = (trimap_tensor[0] == 2) #background
        segment_labels[2, :] = (trimap_tensor[0] == 3) #boundary

        return image_tensor, segment_labels

    def split_dataset(self, train_split, val_split, test_split, batch_size):
        """
        Reads in Oxford IIIT-pet images and splits in training, validation, test data loaders
        """
        # Load the dataset using ImageFolder
        # Calculate the sizes for train, validation, and test sets
        num_samples = len(self)
        num_train = int(train_split * num_samples)
        num_val = int(val_split * num_samples)
        num_test = num_samples - num_train - num_val

        # Shuffle indices and split into training, validation, and test sets
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Define samplers for each split
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        # Create data loaders for each set
        train_loader = DataLoader(self, batch_size = batch_size, sampler=train_sampler, drop_last = True)
        val_loader = DataLoader(self, batch_size = 4, sampler=val_sampler)
        test_loader = DataLoader(self, batch_size = 4, sampler=test_sampler)

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
        classified = self.decoder(encoded)
        return classified



class SegmentationClassifier(nn.Module):
    """
    Takes
        image
    Returns
        class probability per channel-pixel
    """
    def __init__(self, params, num_classes):
        self.image_size = params['image_size']
        self.in_channels = params['num_channels']
        self.out_channels = params['CNN_channels']
        self.kernel_size = params['CNN_kernel']
        super(SegmentationClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=1)
        self.upsample = nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(in_channels=self.out_channels, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv3(x)

        return x




#########
# Control
run_pretraining = True
check_masking = False
check_infilling = False
check_oxford_batch = False
run_semantic_training = True
check_semantic_segmentation = False
save_models = True
load_models = False

if __name__ == '__main__':


    torch.manual_seed(42) # Set seed for random number generator
    device = get_optimal_device()
    #############
    # Definitions
    params = {
        # image
        "image_size": 224,              # number of pixels square
        "num_channels": 3,              # RGB image -> 3 channels
        "patch_size": 16,               # must be divisor of image_size
        # vision transformer encoder
        "vit_num_features": 768,        # 768 number of features created by the vision transformer
        "vit_num_layers": 12,            # 12ViT parameter
        "vit_num_heads": 8,             # 8 ViT parameter
        "vit_hidden_dim": 512,          # 512 ViT parameter
        "vit_mlp_dim": 1024,             # 1024 ViT parameter
        # vision transformer decoder
        "decoder_hidden_dim": 1024,    # 1024 ViT decoder first hidden layer dimension
        "decoder_CNN_channels": 16,    #
        "decoder_scale_factor": 4,     #
        # segmentation model
        "segmenter_hidden_dim": 128,
        "segmenter_classes": 3,         # image, background, boundary
        }

    # image and masking
    mask_ratio = 0.5
    if params['image_size'] % params['patch_size'] !=0: print("Alert! Patch size does not divide image size")
    num_patches = (params['image_size'] // params['patch_size']) **2
    num_masks = int(num_patches * mask_ratio)
    # training
    pt_batch_size=8
    report_every=100
    pt_num_epochs=12
    pt_learning_rate = 0.1
    pt_momentum = 0.9  # not used, since using Adam optimizer
    pt_step = 2
    pt_gamma = 0.3
    # file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir+"/Animals-10/raw-img/"
    model_file = "/masked_autoencoder_model.pth"
    encoder_file = "/masked_autoencoder_encoder.pth"
    decoder_file = "/masked_autoencoder_decoder.pth"
    segmentation_model_file = "/semantic_segmentation_model.pth"
    segmentation_encoder_file = "/segmentation_encoder.pth"
    segmentation_decoder_file = "/segmentation_decoder.pth"
    oxford_dir = "/Oxford-IIIT-Pet_split/images"
    animals_classes = ('cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo') # not used, since pre-training is (unsupervised) variational auto-encoder
    oxford_classes = ('Abyssinian_cat', 'american_bulldog_dog', 'american_pit_bull_terrier_dog', 'basset_hound_dog', 'beagle_dog',
                      'Bengal_cat', 'Birman_cat', 'Bombay_cat', 'boxer_dog', 'British_Shorthair_cat', 'chihuahua_dog', 'Egyptian_Mau_cat',
                      'english_cocker_spaniel_dog', 'english_setter_dog', 'german_shorthaired_dog', 'great_pyrenees_dog', 'havanese_dog',
                      'japanese_chin_dog', 'keeshond_dog', 'leonberger_dog', 'Maine_Coon_cat', 'miniature_pinscher_dog', 'newfoundland_dog',
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
    ft_num_epochs = 30
    ft_lr = 0.1
    ft_step = 5
    ft_gamma = 0.2

    # Instantiate the encoder & decoder for color images, patchmaker and model (encoder/decoder)
    encoder = VisionTransformerEncoder(params) # num_classes == number of features in ViT image embedding
    decoder = VisionTransformerDecoder(params)
    patch_masker = PatchMasker(params['patch_size'], num_masks)
    pt_model = MaskedAutoencoder(encoder, decoder).to(device)

    # if a model has already been saved, load it
    if load_models and os.path.isfile(script_dir+model_file):
        print("Loading pre-saved vision-transformer model")
        encoder.load_state_dict(torch.load(script_dir+encoder_file), strict=False)
        decoder.load_state_dict(torch.load(script_dir+decoder_file), strict=False)
        pt_model.load_state_dict(torch.load(script_dir+model_file), strict=False)

    #####################
    # Demonstrate masking
    if check_masking:
        original_image_tensor, masked_image_tensor = mask_tester(patch_masker, data_dir+"/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")

    ###############
    # MVAE training
    if run_pretraining: # Run training
        print("In pre-training")
        start_time=time.time()
        pt_model.train()

        # Define loss function and optimizer
        pt_criterion = nn.MSELoss()
        pt_optimizer = torch.optim.Adam([{'params': pt_model.encoder.parameters()},
                                         {'params': pt_model.decoder.parameters()}],
                                         lr=pt_learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(pt_optimizer, step_size=pt_step, gamma=pt_gamma)
        # load and pre-process Animals-10 dataset and dataloader & transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = Animals10Dataset(data_dir, (params['image_size'], params['image_size']), transform=transform)
        dataloader = DataLoader(dataset, batch_size=pt_batch_size, shuffle=True, drop_last=True) # drop last batch so that all batches are complete

        # Main training loop
        losses=[]
        for epoch in range(pt_num_epochs):
            epoch_start_time=time.time()
            running_loss = 0.0
            for its, (input_images, input_labels) in enumerate(dataloader):
                input_images, input_labels = input_images.to(device), input_labels.to(device)
                # Add random masking to the input images
                masked_images = patch_masker.mask_patches(input_images)

                # Forward pass & compute the loss
                output_tensors = pt_model(masked_images)
                loss = pt_criterion(output_tensors, input_images)

                # Backward pass and optimization
                pt_optimizer.zero_grad()
                loss.backward()
                pt_optimizer.step()

                running_loss += loss.detach().cpu().item()
                if its % report_every == (report_every-1):    # print every report_every mini-batches
                    print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], running loss: %.4f'
                          % (epoch + 1, pt_num_epochs, pt_batch_size, its + 1, len(dataloader),
                             running_loss / len(dataloader)))
            scheduler.step()
            epoch_end_time = time.time()
            print(f"Epoch [{epoch + 1}/{pt_num_epochs}] completed in {(epoch_end_time-epoch_start_time): .0f}s, "
                  f"Loss: {running_loss / len(dataloader): .4f}")
            losses.append(running_loss / len(dataloader))
        print(f"Masked VAE training finished after {(epoch_end_time-start_time):.0f}s")

        # Save the trained model & losses
        if save_models:
            torch.save(pt_model.state_dict(), script_dir+model_file)
            torch.save(encoder.state_dict(), script_dir+encoder_file)
            torch.save(decoder.state_dict(), script_dir+decoder_file)
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        with open(script_dir+"/pt_losses"+date_str+".txt", 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')
        print("Models saved\nFinished")

    ################################
    # Demonstrate infilling an image
    if check_infilling:
        original_image_tensor, masked_image_tensor = mask_tester(patch_masker,
                                                                 data_dir + "/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
        pt_model.eval()
        with torch.no_grad():
            infill_image_tensor = pt_model(masked_image_tensor.to(device))
            infill_image_tensor = infill_image_tensor.reshape(params['num_channels'], params['image_size'],
                                                              params['image_size'])
            inpainted_image_tensor = masked_image_tensor[0].to(device) + infill_image_tensor.to(device) * \
                                     (masked_image_tensor[0].to(device) == 0).float()

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
        plt.imshow(infill_image_tensor.cpu().permute(1, 2, 0))  # Assuming inpainted_image_tensor is in CHW format
        plt.title('Infill Image')
        #
        plt.subplot(1, 4, 4)
        plt.axis('off')
        plt.imshow(inpainted_image_tensor.cpu().permute(1, 2, 0))  # Assuming inpainted_image_tensor is in CHW format
        plt.title('Inpainted Image')
        #
        plt.show()
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        plt.savefig(script_dir+'/infilling'+date_str+'.png')
        #
        print("Infilling finished")

    # Init model
    encoder = VisionTransformerEncoder(params)
    segmentation_decoder = VisionTransformerDecoder(params)
    ft_segmentation_model = SemanticSegmenter(encoder, segmentation_decoder).to(device)

    if load_models and os.path.isfile(script_dir+segmentation_decoder_file):
        print("Loading pre-saved segmentation model")
        segmentation_decoder.load_state_dict(torch.load(script_dir+segmentation_decoder_file), strict=False)
        encoder.load_state_dict(torch.load(script_dir+encoder_file), strict=False)
        ft_segmentation_model.load_state_dict(torch.load(script_dir+segmentation_model_file), strict=False)

    #############################
    # train semantic segmentation
    if run_semantic_training:
        print("In semantic segmentation training") # images need to be in one folder per class
        start_time=time.time()

        # load data
        oxford_dataset=OxfordPetDataset(image_dir=script_dir+'/Oxford-IIIT-Pet/images/',
                                        label_dir=script_dir+'/Oxford-IIIT-Pet/annotations/trimaps/',
                                        parameters=params)
        train_loader, val_loader, test_loader = oxford_dataset.split_dataset(train_size,val_size,test_size,ft_batch_size)
        if check_oxford_batch:
            view_batch(train_loader)

        # Define loss function and optimizer
        ft_criterion = nn.CrossEntropyLoss()
        ft_optimizer = torch.optim.Adam([{'params': ft_segmentation_model.encoder.parameters()},
                                         {'params': ft_segmentation_model.decoder.parameters()}],
                                         lr=ft_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(ft_optimizer, step_size=ft_step, gamma=ft_gamma)

        losses=[]
        for epoch in range(ft_num_epochs):
            epoch_start_time=time.time()
            running_loss = 0
            for its, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # forward pass
                outputs = ft_segmentation_model(images)
                loss = ft_criterion(outputs, torch.argmax(labels, dim=1))

                # backward pass
                ft_optimizer.zero_grad()
                loss.backward()
                ft_optimizer.step()
                # print(its, loss.detach().cpu().item())
                running_loss += loss.detach().cpu().item()
                if its % report_every == (report_every-1):    # print every report_every mini-batches
                    print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], running loss: %.4f' %
                          (epoch + 1, ft_num_epochs, ft_batch_size, its + 1, len(train_loader),
                           running_loss / len(train_loader)))
            scheduler.step()
            epoch_end_time=time.time()
            losses.append(running_loss / len(train_loader))
            print(f"Epoch [{epoch + 1}/{ft_num_epochs}] completed in {(epoch_end_time-epoch_start_time):.0f}s, Loss: {running_loss / len(train_loader):.4f}")
            view_training(ft_segmentation_model, val_loader)
        print(f"Segmentation training finished after {(epoch_end_time-start_time):.0f}s")

        # save the trained model and losses
        if save_models:
            torch.save(encoder.state_dict(), script_dir+segmentation_encoder_file)
            torch.save(segmentation_decoder.state_dict(), script_dir+segmentation_decoder_file)
            torch.save(ft_segmentation_model.state_dict(), script_dir+segmentation_model_file)
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        with open(script_dir+"/ft_losses"+date_str+".txt", 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')
        print("Fine tune models saved\nFinished")

    ###################################
    # demonstrate semantic segmentation
    if check_semantic_segmentation:
        print("Test semantic segmentation")
        view_training(ft_segmentation_model, test_loader)

    print("Semantic segmentation finished")

    print("Script finished")
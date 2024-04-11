import torch
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import random

image_to_tensor = transforms.ToTensor()
tensor_to_image = transforms.ToPILImage()


def get_optimal_device():
    """
    Returns: optimal device.
        CUDA (GPU) if available, MPS if mac, CPU
    """
    if torch.cuda.is_available():
        ret = "cuda"
    elif torch.backends.mps.is_available():
        ret = "mps"
    else:
        ret = "cpu"
    print(f"Running on {ret}")
    return ret


def get_optimizer(model, params):
    match params['optimizer']:
        case "Adam":
            return optim.Adam(model.parameters(), lr=params['learning_rate']) 
        case "AdamW":
            return optim.AdamW(model.parameters(), lr=params['learning_rate']) 
        case "SGD":
            return optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'])
        case _:
            print("Optimizer nor known")


def get_network(params, output_channels):
    match params['network']:
        case "CNN":
            return CNNEncoder(params), CNNDecoder(params, output_channels)
        case "ViT":
            return VisionTransformerEncoder(params), CNNDecoder(params, output_channels)
        case "Linear":
            return LinearEncoder(params), CNNDecoder(params, output_channels)
        case _:
            print("Network not known")


def view_training(model, loader, display):
    images, labels = next(iter(loader))
    num_images = min(images.size(0),4)

    outputs = model(images.to(device))
    outputs = outputs.cpu().detach()
    images, labels = images.cpu(), labels.cpu()
    output_labels = torch.argmax(outputs.cpu().detach(), dim=1)

    if display is False:
        plt.ioff()
    fig, axes = plt.subplots(4, num_images, figsize=(3*num_images,8))
    time.sleep(1)
    for i in range(num_images):
        if num_images>1:
            ax0 = axes[0,i]
            ax1 = axes[1,i]
            ax2 = axes[2,i]
            ax3 = axes[3,i]
        else:
            ax0 = axes[0]
            ax1 = axes[1]
            ax2 = axes[2]
            ax3 = axes[3]
        ax0.axis('off')
        ax0.set_title('Image')
        ax0.imshow(images[i].permute(1,2,0))
        ax1.axis('off')
        ax1.set_title('Label')
        ax1.imshow(labels[i].permute(1,2,0))
        ax2.axis('off')
        ax2.set_title('Output (prob)')
        ax2.imshow(outputs[i].permute(1,2,0))
        ax3.axis('off')
        ax3.set_title('Output (argmax)')
        ax3.imshow(output_labels[i])
    plt.tight_layout()
    plt.show()
    date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
    plt.savefig(params['script_dir']+'/Output/labels'+date_str+'.png')
    plt.close()


def overlap(model, loader):
    images, labels = next(iter(loader))
    outputs = model(images.to(device))
    outputs = outputs.cpu().detach()
    images, labels = images.cpu(), labels.cpu()
    output_labels = torch.argmax(outputs.cpu().detach(), dim=1)
    overlap = labels == output_labels
    overlap_fraction = overlap.float().mean().item()

    return overlap_fraction


def mask_tester(patch_masker, model, loader, display):
    """
    Takes
        patch masker object, link to image file
    Displays
        original and patched image
    Returns
        input image, masked image tensor
    """
    images, _ = next(iter(loader))
    num_images = min(images.size(0),4)
    masked_images, masks = patch_masker.mask_patches(images)
    with torch.no_grad():
        infill_images = torch.sigmoid(model(masked_images.to(device)))
        inpainted_images = masked_images.to(device) + infill_images.to(device) * (masked_images.to(device) == 0).float()

    if display is False:
        plt.ioff()
    fig, axes = plt.subplots(4, num_images, figsize=(12, 9))
    time.sleep(1)
    for i in range(num_images):
        if num_images>1:
            ax0 = axes[0,i]
            ax1 = axes[1,i]
            ax2 = axes[2,i]
            ax3 = axes[3,i]
        else:
            ax0 = axes[0]
            ax1 = axes[1]
            ax2 = axes[2]
            ax3 = axes[3]
        ax0.axis('off')
        ax0.set_title('Image')
        ax0.imshow(images[i].permute(1,2,0))
        ax1.set_title('Masked Image')
        ax1.imshow(masked_images[i].permute(1,2,0)) 
        ax2.set_title('Infill')
        ax2.imshow(infill_images[i].cpu().permute(1,2,0)) 
        ax3.set_title('Inpainted')
        ax3.imshow(inpainted_images[i].cpu().permute(1,2,0)) 
    plt.tight_layout()
    plt.show()
    date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
    plt.savefig(params['script_dir']+'/Output/masks'+date_str+'.png')
    plt.close()


def initialise_weights(model):
    """
    Random initialisation of model weights
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'embedding' in name or 'fc' in name:
                nn.init.xavier_uniform_(param)
            elif 'conv' in name:  # Initialize convolutional layers
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)  # Default initialization for other layers
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)  # Initialize biases to zero


def save_losses(losses, stage, params):
    date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
    with open(params['script_dir']+"/Output/"+stage+"_losses"+date_str+".txt", 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        for i, loss in enumerate(losses):
            f.write(f'{i+1}  {loss}\n')


class CNNEncoder(nn.Module):
    def __init__(self, params):
        super(CNNEncoder, self).__init__()
        self.num_features = params['num_features']
        self.num_channels = params['num_channels']
        self.hidden_dim = params['hidden_dim']
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=8, kernel_size=3, stride=1, padding=1) #in 224, out 112
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1) #in 112 out 56
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) #in 56 out 28
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) #in 28 out 14
        self.fc1 = nn.Linear(64*14*14, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_features)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 64*14*14)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return x #output batch * num_features output


class CNNDecoder(nn.Module):
    """
        I've made output channels an explicit parameter to enable the same code to be used both for pixel generation
        (in which case the outputs are the 3 primary colours) and for pixel classification (in which case the outputs
        are the number of classes, which in the pet dataset happens to be 3, but this is coincidental)
    """
    def __init__(self, params, output_channels):
        super(CNNDecoder, self).__init__()
        self.num_features = params['num_features']
        self.output_channels = output_channels
        self.fc = nn.Linear(self.num_features, 512 * 14 * 14)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(64, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.bnc1 = nn.BatchNorm2d(256)
        self.bnc2 = nn.BatchNorm2d(128)
        self.bnc3 = nn.BatchNorm2d(64)
        self.bnc4 = nn.BatchNorm2d(self.output_channels)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 14, 14)
        x = self.bnc1(self.relu(self.conv1(x))) # output 28
        x = self.bnc2(self.relu(self.conv2(x))) # output 56
        x = self.bnc3(self.relu(self.conv3(x))) # output 112
        x = torch.sigmoid(self.bnc4(self.conv4(x))) #output 224 * 224
        #x = torch.softmax(x, dim=1) # normalise to [0,1]

        return x # output batch * output_channels * image_size * image_size


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
        self.num_features = params['num_features']
        self.num_layers = params['vit_num_layers']
        self.num_heads = params['vit_num_heads']
        self.hidden_dim = params['hidden_dim']
        self.mlp_dim = params['vit_mlp_dim']
        self.vit = models.vision_transformer.VisionTransformer(image_size = self.image_size,
                                                               patch_size = self.patch_size,
                                                               num_classes = self.num_features,
                                                               num_layers = self.num_layers,
                                                               num_heads = self.num_heads,
                                                               hidden_dim = self.hidden_dim,
                                                               mlp_dim = self.mlp_dim,
                                                               dropout=0.1)

    def forward(self, x):
        # Pass the input through the ViT-B_16 backbone
        features = self.vit(x)
        return features


class LinearEncoder(nn.Module):
    def __init__(self, params):
        super(LinearEncoder, self).__init__()
        self.output_dim = params['num_features']
        self.hidden_dim = params['hidden_dim']
        self.image_size = params['image_size']
        self.num_channels = params['num_channels']
        self.fc1= nn.Linear(self.num_channels*self.image_size*self.image_size, self.hidden_dim)
        self.fc2= nn.Linear(self.hidden_dim, self.output_dim) 
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class SegmentModel(nn.Module):
    """
    Takes
        batch of images
    Returns
        either:
         - class probability per channel-pixel
         - value of each color pixel
    """
    def __init__(self, encoder, decoder):
        super(SegmentModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x) # image to features
        decoded = self.decoder(encoded) # features to segmentation map
        return decoded


class OxfordPetDataset(Dataset):
    """
    Takes
        links to jpg images and trimap pngs
    Returns
        image tensors and classification map
    """
    def __init__(self, original_dataset, params):
        self.original_dataset = original_dataset
        self.params = params

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, target = self.original_dataset[idx]

        target_transform = transforms.Compose([
            transforms.Resize((self.params['image_size'], self.params['image_size'])),  # Resize the input PIL Image to given size.
            transforms.ToTensor(),                                            # Convert a PIL Image to PyTorch tensor.
            transforms.Lambda(lambda x: ((x*255)-1).long())])                 # Convert back to class values 0,1,2
        target = target_transform(target)

        return image, target


class Animals10Dataset(Dataset):
    """
    Loads and cleans up images in Animals-10 dataset
    """
    def __init__(self, params):
        self.root_dir = params['script_dir']+"/Data/Animals-10/raw-img"
        self.params = params,
        self.target_size = params['image_size']
        self.transform = transforms.Compose([
             transforms.Resize((self.target_size, self.target_size)),
             transforms.ToTensor(),
             # transforms.Normalize(mean=[0.45, 0.5, 0.55], std=[0.2, 0.2, 0.2]) # normalising helps convergence
             ])
        self.classes = sorted(os.listdir(self.root_dir))
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
        
        img = img.resize((self.target_size, self.target_size))
        # img = transforms.ToTensor()(img)

        img = img.resize((self.target_size, self.target_size))  # Resize the image
        if self.transform:
            img = self.transform(img)
        return img, label


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
        masks = torch.ones_like(masked_image)
        for index in masked_patch_indices:
            i, j = index
            y_start = i * self.patch_size
            y_end = min((i + 1) * self.patch_size, height)
            x_start = j * self.patch_size
            x_end = min((j + 1) * self.patch_size, width)
            masked_image[:, :, y_start:y_end, x_start:x_end] = 0  # Set masked patch to zero
            masks[:, :, y_start:y_end, x_start:x_end] = 0 

        return masked_image, masks




if __name__ == '__main__':

    ##
    ## Setup
    ##

    torch.manual_seed(42) # Set seed for random number generator
    device = get_optimal_device()

    control_params = {
        "run_pretrainer": False,
        "check_masking": False,
        "check_infilling": False,
        "check_oxford_batch": False,
        "run_finetuner": True,
        "check_semantic_segmentation": True,
        "save_models": False,
        "load_models": False,
        "encoder_file": "/Models/encoder.pth",
        "decoder_file": "/Models/masked_autoencoder_decoder.pth",
        "segmentation_decoder_file": "/Models/segmentation_decoder.pth"
    }
    image_params = {
        "image_size": 224,              # number of pixels square
        "num_channels": 3,              # RGB image -> 3 channels
        "patch_size": 14,               # must be divisor of image_size
        'batch_size': 32,
        'num_classes': 3,
        'mask_ratio': 0.25,
    }
    network_params = {
        'network': "CNN",               # CNN, ViT, Linear
        'num_features': 768,            # 768
        'hidden_dim': 2048,
        "vit_num_layers": 4,            # 12ViT parameter
        "vit_num_heads": 8,             # 8 ViT parameter
        "vit_mlp_dim": 2048,            # 1024 ViT parameter
    }
    training_params = {
        'optimizer': "Adam",            # Adam, AdamW, SGD
        'pt_num_epochs': 16,
        'ft_num_epochs': 5,
        'learning_rate': 0.001,
        'momentum': 0.9,                # not used in Adam
        'report_every': 10,
        'class_weights': [1.0,0.5,1.5], # pet, background, boundary
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    control_params['script_dir'] = script_dir
    # image and masking
    if image_params['image_size'] % image_params['patch_size'] !=0: print("Alert! Patch size does not divide image size")
    num_patches = (image_params['image_size'] // image_params['patch_size']) **2
    num_masks = int(num_patches * image_params['mask_ratio'])
    image_params['num_patches'] = num_patches
    image_params['num_masks'] = num_masks

    params={**control_params, ** image_params, **network_params, **training_params}

    ##
    ## Pre trainer
    ##
    if params['run_pretrainer']: # Run training
        print("In pre-training")
        start_time=time.time()

        # load and pre-process Animals-10 dataset and dataloader & transform to normalize the data
        transform = transforms.Compose([transforms.ToTensor()])
        pt_dataset = Animals10Dataset(params)
        pt_dataloader = DataLoader(pt_dataset, batch_size = params['batch_size'], shuffle = True, drop_last = True) # drop last batch so that all batches are complete


        # Instantiate the encoder & decoder for color images, patchmaker and model (encoder/decoder)
        encoder , pt_decoder = get_network(params, params['num_channels'])
        if params['load_models'] and os.path.isfile(params['script_dir']+params['encoder_file']):
            encoder.load_state_dict(torch.load(params['script_dir']+params['encoder_file']), strict=False)
        else:
            initialise_weights(encoder)
        if params['load_models'] and os.path.isfile(params['script_dir']+params['decoder_file']):
            pt_decoder.load_state_dict(torch.load(params['script_dir']+params['decoder_file']), strict=False)
        else:
            initialise_weights(pt_decoder)
        vae_model = SegmentModel(encoder, pt_decoder).to(device) # vae pre-trainer and supervised fine-tuner share encoder
        patch_masker = PatchMasker(params['patch_size'], params['num_masks'])


        # test everything is working
        print("View images, masked imaged and predicted images before starting")
        mask_tester(patch_masker, vae_model, pt_dataloader, True)


        pt_criterion = nn.MSELoss()
        pt_optimizer = get_optimizer(vae_model, params)


        # Main training loop
        losses=[]
        for epoch in range(params['pt_num_epochs']):
            epoch_start_time=time.time()
            running_loss = 0.0
            for its, (input_images, input_labels) in enumerate(pt_dataloader):
                input_images, input_labels = input_images.to(device), input_labels.to(device)
                # Add random masking to the input images
                masked_images, masks = patch_masker.mask_patches(input_images)

                # Forward pass & compute the loss
                logits = vae_model(masked_images)
                outputs = torch.sigmoid(logits) # squash to 0-1 pixel values
                masked_outputs = outputs * masks #dont calculate loss for masked portion
                loss = pt_criterion(masked_outputs, masked_images) / (1.0-params['mask_ratio']) # normalise to make losses comparable across different mask ratios

                # Backward pass and optimization
                pt_optimizer.zero_grad()
                loss.backward()
                pt_optimizer.step()

                running_loss += loss.detach().cpu().item()
                if its % params['report_every'] == (params['report_every']-1):    # print every report_every mini-batches
                    print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], running loss: %.4f' % (epoch + 1, params['pt_num_epochs'], params['batch_size'], its + 1, len(pt_dataloader), running_loss / its))
            #scheduler.step()
            epoch_end_time=time.time()
            print(f"Epoch [{epoch + 1}/{params['pt_num_epochs']}] completed in {(epoch_end_time-epoch_start_time):.0f}s, Loss: {(running_loss / its):.4f}")
            losses.append(running_loss)
        print(f"Masked VAE training finished after {(epoch_end_time-start_time):.0f}s")


        # Save the trained model & losses
        if params['save_models']:
            torch.save(encoder.state_dict(), params['script_dir']+params['encoder_file'])
            torch.save(pt_decoder.state_dict(), params['script_dir']+params['decoder_file'])
        save_losses(losses, "pt", params)
        plt.plot(losses)
        plt.title("Pre-trainer losses")
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        plt.savefig(params['script_dir']+'/Output/pt_losses'+date_str+'.png')
        plt.show()
        plt.close()
        mask_tester(patch_masker, vae_model, pt_dataloader, True)





    ##
    ## Fine tuner
    ##
    if params['run_finetuner']: # Run training
        print("In fine-tuning")
        start_time=time.time()

        # Download, transform and load the dataset
        transform = transforms.Compose([transforms.Resize((params['image_size'], params['image_size'])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.45, 0.5, 0.55], std=[0.2, 0.2, 0.2]) # normalising helps convergence
                                        ]) # Define data transformations: resize and convert to PyTorch tensors
        train_dataset = datasets.OxfordIIITPet(root=params['script_dir']+"/Data/OxfordIIITPet/Train", split='trainval', download=True, target_types='segmentation', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params['batch_size'], shuffle=True)
        test_dataset = datasets.OxfordIIITPet(root=params['script_dir']+"/Data/OxfordIIITPet/Test", split='test', download=True, target_types='segmentation', transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True)


        # initialize model: encoder and decoder
        try:
            encoder # if already exists (from pre-trainer) don't re-initialise
        except NameError:
            encoder , decoder = get_network(params, params['num_classes'])
            if params['load_models'] and os.path.isfile(params['script_dir']+params['encoder_file']):
                encoder.load_state_dict(torch.load(params['script_dir']+params['encoder_file']), strict=False)
            else:
                initialise_weights(encoder)
        else:
            print("encoder exists")
            _ , decoder = get_network(params, params['num_classes']) # even thought networks are the same, image decoder and image classifier are different instances
        if params['load_models'] and os.path.isfile(params['script_dir']+params['segmentation_decoder_file']):
            decoder.load_state_dict(torch.load(params['script_dir']+params['segmentation_decoder_file']), strict=False)
        else:
            initialise_weights(decoder)
        segment_model = SegmentModel(encoder, decoder).to(device)



        # Initialize dataset & dataLoader
        dataset = OxfordPetDataset(train_dataset, params)
        training_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)


        # test everything is working
        print("View images, labels and as yet unlearned model output before starting")
        view_training(segment_model, training_loader, True)
        print(f"Starting overlap: {overlap(segment_model, training_loader):.3f}")

        ## loss and optimiser
        class_weights = torch.tensor(params['class_weights']).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device) # use of weights to correct for disparity in foreground, background, boundary
        optimizer = get_optimizer(segment_model, params)


        ## train
        losses=[]
        for epoch in range(params['ft_num_epochs']):  # loop over the dataset multiple times
            epoch_start_time=time.time()

            running_loss = 0.0
            for its, (inputs, labels) in enumerate(training_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze(1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = segment_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.detach().cpu().item()
                if its % params['report_every'] == (params['report_every']-1):    # print every report_every mini-batches
                    print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], running loss: %.4f' %
                            (epoch + 1, params['ft_num_epochs'], params['batch_size'], its + 1, len(train_loader), running_loss / its))
            # end its
            losses.append(running_loss)
            epoch_end_time=time.time()
            end_time_str = time.strftime("%H.%M", time.localtime(epoch_end_time))
            print(f"Epoch [{epoch + 1}/{params['ft_num_epochs']}] completed at {end_time_str} taking {(epoch_end_time-epoch_start_time):.0f}s, Current LR {optimizer.param_groups[0]['lr']:.7f}, Loss: {(running_loss / its):.4f}, Sample pixel overlap: {overlap(segment_model, training_loader):.3f}")
        #end epochs

        if params['save_models']:
            torch.save(encoder.state_dict(), params['script_dir']+params['encoder_file'])
            torch.save(decoder.state_dict(), params['script_dir']+params['segmentation_decoder_file'])
        save_losses(losses, "ft", params)
        plt.plot(losses)
        plt.title("Fine-tuner losses")
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        plt.savefig(params['script_dir']+'/Output/ft_losses'+date_str+'.png')
        plt.show()
        plt.close()
        view_training(segment_model, training_loader, True) # dont understand why this doesnt display


    #display inference on test set
    testing_dataset = OxfordPetDataset(test_dataset, params)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=params['batch_size'], shuffle=True)
    for its in range(5):
        view_training(segment_model, test_loader, True)
        print(f"Sample test set overlap: {overlap(segment_model, test_loader):.3f}")


    print('Finished')
import sys
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

# our code
from src.utils.paths import *
from src.utils.device import get_optimal_device
from src.mvae.data_handler import (
    Animals10Dataset,
    PatchMasker,
    test_patch_masker
)
from src.shared_network_architectures.networks_pt import (
    MaskedAutoencoder,
    VisionTransformerEncoder,
    VisionTransformerDecoder
)

## Control
## Training
run_pretraining = True
check_masking = False
check_infilling = True
save_models = True
load_models = False

## Testing
# run_pretraining = False
# check_masking = True
# check_infilling = True
# save_models = False
# load_models = True

# Parameters
params = {
    # Image
    "image_size": 224,  # number of pixels square
    "num_channels": 3,  # RGB image -> 3 channels
    "patch_size": 16,  # must be divisor of image_size

    # Vision Transformer Encoder
    "vit_num_features": 768,  # 768 number of features created by the vision transformer
    "vit_num_layers": 12,  # 12 ViT parameter
    "vit_num_heads": 8,  # 8 ViT parameter
    "vit_hidden_dim": 1024,  # 1024 ViT parameter
    "vit_mlp_dim": 2048,  # 1024 ViT parameter

    # Vision Transformer Decoder
    "decoder_hidden_dim": 1024,  # 1024 ViT decoder first hidden layer dimension
    "decoder_CNN_channels": 16,
    "decoder_scale_factor": 4,

    # Segmentation Model
    "segmenter_hidden_dim": 128,
    "segmenter_classes": 3,  # image, background, boundary

    # Training
    "mask_ratio": 0.5,
    "pt_batch_size": 8,
    "pt_learning_rate": 0.0001,  # 0.00001
    "pt_num_epochs" : 1,
    # pt_momentum = 0.9  # not used, since using Adam optimizer
    "pt_step": 2,
    "pt_gamma": 0.3
}

wandb.init(project="mvae", entity="adl_team_grey", config=params)

report_every = 100

# Hyper-parameters
mask_ratio = params['mask_ratio']
pt_batch_size = params['pt_batch_size']
pt_num_epochs = params['pt_num_epochs']
pt_learning_rate = params['pt_learning_rate']
# pt_momentum = 0.9  # not used, since using Adam optimizer
pt_step = params['pt_step']
pt_gamma = params['pt_gamma']

# file paths
data_dir = os.path.join(datasets_dir,"Animals-10/raw-img/")
model_file = "masked_autoencoder_model.pth"
encoder_file = "masked_autoencoder_encoder.pth"
decoder_file = "masked_autoencoder_decoder.pth"

# test image
test_image_path = os.path.join(data_dir, "image_0.png")

if __name__ == '__main__':

    # Set seeds for random number generator
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model_path = os.path.join(models_dir, model_file)
    encoder_path = os.path.join(models_dir, encoder_file)
    decoder_path = os.path.join(models_dir, decoder_file)

    print(f"{data_dir = }\n{model_path = }\n{encoder_path = }\n{decoder_path = }\n{test_image_path = }")

    device = get_optimal_device()

    if params['image_size'] % params['patch_size'] != 0:
        raise ValueError("Alert! Patch size does not divide image size")

    num_patches = (params['image_size'] // params['patch_size']) ** 2
    num_masks = int(num_patches * mask_ratio)

    # Instantiate the encoder & decoder for color images, patchmaker and model (encoder/decoder)
    encoder = VisionTransformerEncoder(params)  # num_classes == number of features in ViT image embedding
    decoder = VisionTransformerDecoder(params)
    patch_masker = PatchMasker(params['patch_size'], num_masks)
    pt_model = MaskedAutoencoder(encoder, decoder).to(device)

    #####################
    # if a model has already been saved, load
    if load_models and os.path.isfile(model_path):
        print(f"Loading pre-saved vision-transformer model {model_file}")
        encoder.load_state_dict(torch.load(encoder_path), strict=False)
        decoder.load_state_dict(torch.load(decoder_path), strict=False)
        pt_model.load_state_dict(torch.load(model_path), strict=False)

    #####################
    # Demonstrate masking
    if check_masking:
        # original_image_tensor, masked_image_tensor = mask_tester(patch_masker, data_dir+"/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
        original_image_tensor, masked_image_tensor = test_patch_masker(patch_masker, test_image_path)

    ###############
    #  mvae training
    if run_pretraining:
        print("In pre-training")
        start_time = time.perf_counter()
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

        dataloader = DataLoader(dataset,
                                batch_size=pt_batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=2)  # drop last batch so that all batches are complete

        # Main training loop
        losses = []
        for epoch in range(pt_num_epochs):
            epoch_start_time = time.perf_counter()
            running_loss = 0.0
            for its, input_images in enumerate(dataloader):
                input_images = input_images.to(device)
                masked_images = patch_masker.mask_patches(input_images)

                # Forward pass & compute the loss
                output_tensors = pt_model(masked_images)
                loss = pt_criterion(output_tensors, input_images)
                
                # Backward pass and optimization
                pt_optimizer.zero_grad()
                loss.backward()
                pt_optimizer.step()

                # Calculate average loss
                average_loss = running_loss / (its + 1)

                # Log metrics to wandb
                wandb.log({"Running Loss": loss.item(), "Average Loss": average_loss, "Epoch": epoch + 1})

                running_loss += loss.detach().cpu().item()

                if its % report_every == (report_every - 1):  # print every report_every mini-batches
                    curr_time = time.perf_counter() - start_time
                    print(f'Epoch [{epoch + 1} / {pt_num_epochs}], {pt_batch_size} image minibatch [{its + 1} / {len(dataloader)}], '
                        f'average loss: {average_loss:.4f}, uptime: {curr_time:.2f}')
                    
            scheduler.step()
            epoch_time = time.perf_counter() - epoch_start_time
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{pt_num_epochs}] completed in {(epoch_time):.0f}s, Loss: {running_loss / len(dataloader):.4f}")
            losses.append(epoch_loss)

            wandb.log({"Epoch Loss": epoch_loss, "Epoch Time": epoch_time})

        end_time = time.perf_counter()
        print(f"Masked VAE training finished after {(end_time - start_time):.0f}s")


        """
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        with open(os.path.join(mvae_dir, "pt_losses" + date_str + ".txt"), 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')
        print("Models saved\nFinished")
        """

    ###############
    # Demonstrate infilling an image
    if check_infilling:
        # original_image_tensor, masked_image_tensor = mask_tester(patch_masker, data_dir+"/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
        original_image_tensor, masked_image_tensor =  test_patch_masker(patch_masker, test_image_path)

        pt_model.eval()
        with torch.no_grad():
            infill_image_tensor = pt_model(masked_image_tensor.to(device))
            infill_image_tensor = infill_image_tensor.reshape(params['num_channels'], params['image_size'],
                                                              params['image_size'])
            inpainted_image_tensor = masked_image_tensor[0].to(device) + infill_image_tensor.to(device) * (
                        masked_image_tensor[0].to(device) == 0).float()

        # Visualize the results (assuming you have matplotlib installed)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 4, 1)
        plt.axis('off')
        plt.imshow(original_image_tensor.permute(1, 2, 0))  # Assuming image_tensor is in CHW format
        plt.title('Original Image')

        plt.subplot(1, 4, 2)
        plt.axis('off')
        plt.imshow(masked_image_tensor[0].permute(1, 2, 0))
        plt.title('Masked image')

        plt.subplot(1, 4, 3)
        plt.axis('off')
        plt.imshow(infill_image_tensor.cpu().permute(1, 2, 0))  # Assuming inpainted_image_tensor is in CHW format
        plt.title('Infill Image')

        plt.subplot(1, 4, 4)
        plt.axis('off')
        plt.imshow(inpainted_image_tensor.cpu().permute(1, 2, 0))  # Assuming inpainted_image_tensor is in CHW format
        plt.title('Inpainted Image')

        plt.show()
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        plt.savefig(os.path.join(mvae_dir,'infilling' + date_str + '.png'))

        wandb.log({"Infilling Results": plt})

        print("Infilling finished")

    print("MVAE Script complete")

    print("Saving Models")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Current timestamp
    final_epoch_loss = epoch_loss  # Replace 'epoch_loss' with the actual value

    # Define model file names
    pt_model_file = f"model_{timestamp}.pt"
    encoder_model_file = f"encoder_model_{timestamp}.pt"
    decoder_model_file = f"decoder_model_{timestamp}.pt"
    epoch_loss_file = f"epoch_loss_{final_epoch_loss}.txt"  # Optionally include epoch loss in the file name

    # Save the trained model & losses
    if save_models:
        pt_model_path = os.path.join(models_dir, pt_model_file)
        torch.save(pt_model.state_dict(), pt_model_path)

        encoder_path = os.path.join(models_dir, encoder_model_file)
        torch.save(encoder.state_dict(), encoder_path)

        decoder_path = os.path.join(models_dir, decoder_model_file)
        torch.save(decoder.state_dict(), decoder_path)

# External packages
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import matplotlib.pyplot as plt

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
check_infilling = False
save_models = True
load_models = False

## Testing
# run_pretraining = False
# check_masking = True
# check_infilling = True
# save_models = False
# load_models = True

## Definitions
params = {
    # image
    "image_size": 224,  # number of pixels square
    "num_channels": 3,  #  RGB image -> 3 channels
    "patch_size": 16,  # must be divisor of image_size

    # vision transformer encoder
    "vit_num_features": 768,  # 768 number of features created by the vision transformer
    "vit_num_layers": 14, #12,  # 12ViT parameter
    "vit_num_heads": 8,  # 8 ViT parameter
    "vit_hidden_dim": 1024, #512,  # 512 ViT parameter
    "vit_mlp_dim": 2048, #1024,  # 1024 ViT parameter

    # vision transformer decoder
    "decoder_hidden_dim": 1024,  # 1024 ViT decoder first hidden layer dimension
    "decoder_CNN_channels": 16,  #
    "decoder_scale_factor": 4,  #

    # segmentation model
    "segmenter_hidden_dim": 128,
    "segmenter_classes": 3,  # image, background, boundary
}

report_every = 100

# Hyper-parameters
mask_ratio = 0.5
pt_batch_size = 8
pt_num_epochs = 3
pt_learning_rate = 0.00001  # 0.00001
# pt_momentum = 0.9  # not used, since using Adam optimizer
pt_step = 2
pt_gamma = 0.3

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
                # print(f"[{its}, {len(dataloader)}] {time.perf_counter() - epoch_start_time:.0f}s")
                input_images = input_images.to(device)

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
                if its % report_every == (report_every - 1):  # print every report_every mini-batches
                    curr_time = time.perf_counter() - start_time
                    print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], cumulative running loss: %.4f, uptime: %.2f' % (
                        epoch + 1, pt_num_epochs, pt_batch_size, its + 1, len(dataloader), running_loss / len(dataloader),
                        curr_time))
            scheduler.step()
            epoch_end_time = time.perf_counter()
            print(
                f"Epoch [{epoch + 1}/{pt_num_epochs}] completed in {(epoch_end_time - epoch_start_time):.0f}s, Loss: {running_loss / len(dataloader):.4f}")
            losses.append(running_loss / len(dataloader))
        end_time = time.perf_counter()
        print(f"Masked VAE training finished after {(end_time - start_time):.0f}s")

        # Save the trained model & losses
        if save_models:
            torch.save(pt_model.state_dict(), model_path)
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)

        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        with open(os.path.join(mvae_dir, "pt_losses" + date_str + ".txt"), 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')
        print("Models saved\nFinished")

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

        print("Infilling finished")

    print("MVAE Script complete")
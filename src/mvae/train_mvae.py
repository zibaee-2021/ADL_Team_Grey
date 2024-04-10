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
from src.IoUMetric import IoULoss

# our code
from src.utils.paths import *
from src.utils.device import get_optimal_device
from src.utils.model_init import initialise_weights
from src.utils.optimizer import get_optimizer
from src.shared_network_architectures.networks_pt import (
    get_network,
    SegmentModel
)
from src.mvae.data_handler import (
    Animals10Dataset,
    PatchMasker
)


## Control
## Training
run_pretraining_and_save = True
check_masking_and_infilling = True
report_every = 100
save_models = run_pretraining_and_save  # If training, then save!
load_models = not run_pretraining_and_save  # Don't load if training but do if not


## Testing
#TODO: need to make it loads models with dates....
# This will need to include a manual insertion of the datestring I think
# run_pretraining_and_save = False
# check_masking_and_infilling = True
# report_every = 100
# save_models = run_pretraining_and_save  # If training, then save!
# load_models = not run_pretraining_and_save  # Don't load if training but do if not


params = {
    # Image
    "image_size": 224,  # number of pixels square
    "num_channels": 3,  #  RGB image -> 3 channels
    "patch_size": 14,  # must be divisor of image_size
    'num_classes': 3,

    # Network
    'network': "CNN",  # CNN, ViT, Linear
    'num_features': 768,  # 768
    'hidden_dim': 2048,
    "vit_num_layers": 4,  # 12ViT parameter
    "vit_num_heads": 8,  # 8 ViT parameter
    "vit_mlp_dim": 2048,  # 1024 ViT parameter

    # Hyper paramaters
    'pt_batch_size': 32,
    'mask_ratio': 0.25,
    'learning_rate': 0.0001,
    'pt_momentum': 0.9,  # not used in Adam

    # Training
    'optimizer': "Adam",  # Adam, AdamW, SGD
    'pt_num_epochs': 8,
}

mask_ratio = params['mask_ratio']
pt_batch_size = params['pt_batch_size']
pt_num_epochs = params['pt_num_epochs']
pt_learning_rate = params['learning_rate']
pt_momentum = params['pt_momentum']
pt_optimizer = params['optimizer']

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

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    model_path = os.path.join(models_dir, model_file)
    encoder_path = os.path.join(models_dir, encoder_file)
    decoder_path = os.path.join(models_dir, decoder_file)

    print(f"{data_dir = }\n{model_path = }\n{encoder_path = }\n{decoder_path = }\n{test_image_path = }")

    device = get_optimal_device()

    if params['image_size'] % params['patch_size'] != 0:
        raise ValueError(f"Patch size ({params['patch_size']}) does not divide image size ({params['image_size']})")

    num_patches = (params['image_size'] // params['patch_size']) ** 2
    num_masks = int(num_patches * mask_ratio)

    # update params
    params['num_patches'] = num_patches
    params['num_masks'] = num_masks

    # Keep track of params if using wandb:
    wandb.init(project="mvae", entity="adl_team_grey", config=params)


    # dataloader and model definition
    # load and pre-process Animals-10 dataset and dataloader & transform to normalize the data
    pt_dataset = Animals10Dataset(root_dir=os.path.join(animals_10_dir, "raw-img"))
    pt_dataloader = DataLoader(pt_dataset,
                               batch_size=params['pt_batch_size'],
                               shuffle=True,
                               drop_last=True,  # drop last batch so that all batches are complete
                               num_workers=2)

    # Instantiate the encoder & decoder for color images, patchmaker and model (encoder/decoder)
    encoder, pt_decoder = get_network(params, params['num_channels'])
    if load_models:
        assert os.path.isfile(encoder_path), f"Expected to laod model but no path exists {encoder_path}"
        # encoder.load_state_dict(torch.load(encoder_path), strict=False)
        encoder.load_state_dict(torch.load(encoder_path))
    else:  # we will train
        initialise_weights(encoder)
    if load_models:
        assert os.path.isfile(decoder_path), f"Expected to laod model but no path exists {decoder_path}"
        pt_decoder.load_state_dict(torch.load(decoder_path), strict=False)
    else:  # we will train
        initialise_weights(pt_decoder)

    vae_model = SegmentModel(encoder, pt_decoder).to(
        device)  # vae pre-trainer and supervised fine-tuner share encoder
    patch_masker = PatchMasker(params['patch_size'], params['num_masks'])

    # test everything is working
    print("View images, masked imaged and predicted images before starting")
    patch_masker.test(vae_model, pt_dataloader, True, device)

    ###############
    #  mvae training
    vae_model.train()
    if run_pretraining_and_save:
        print("In pre-training")
        start_time = time.perf_counter()

        loss_func_choice = {'mse': nn.MSELoss(),
                                'cel': nn.CrossEntropyLoss(),
                                'iou': IoULoss(preds_are_logits=False).forward}
        pt_criterion = loss_func_choice['mse']
        # pt_criterion = nn.MSELoss()  # this was here before

        pt_optimizer = get_optimizer(vae_model, params)

        # Main training loop
        losses = []
        for epoch in range(params['pt_num_epochs']):
            epoch_start_time = time.perf_counter()
            running_loss = 0.0

            for its, input_images in enumerate(pt_dataloader):
                # # TODO Remove - just to speed up testing
                # if its > 101:
                #     break
                input_images = input_images.to(device)
                masked_images, masks = patch_masker.mask_patches(input_images)

                # Forward pass & compute the loss
                outputs = vae_model(masked_images)
                outputs = torch.softmax(outputs, dim=1)  #  squash to 0-1 pixel values

                # TODO: CHECK THE BELOW, WHERE HOULD WE BE CALCULATING THE LOSS
                masked_outputs = outputs * masks  # dont calculate loss for masked portion
                loss = pt_criterion(masked_outputs, input_images)

                # Backward pass and optimization
                pt_optimizer.zero_grad()
                loss.backward()
                pt_optimizer.step()

                # Calculate average loss
                average_loss = running_loss / (its + 1)
                running_loss += loss.detach().cpu().item()

                # Log metrics to wandb
                wandb.log({"Running Loss": loss.item(), "Average Loss": average_loss, "Epoch": epoch + 1})

                if its % report_every == (report_every - 1):  # print every report_every mini-batches
                    curr_time = time.perf_counter() - start_time
                    print(
                        f'Epoch [{epoch + 1} / {pt_num_epochs}], {pt_batch_size} image minibatch [{its + 1} / {len(pt_dataloader)}], '
                        f'average loss: {average_loss:.4f}, uptime: {curr_time:.2f}')

            epoch_time = time.perf_counter() - epoch_start_time
            epoch_loss = running_loss / len(pt_dataloader)
            print(
                f"Epoch [{epoch + 1}/{pt_num_epochs}] completed in {(epoch_time):.0f}s, Loss: {running_loss / len(pt_dataloader):.4f}")
            losses.append(epoch_loss)

            wandb.log({"Epoch Loss": epoch_loss, "Epoch Time": epoch_time})

        end_time = time.perf_counter()
        print(f"Masked VAE training finished after {(end_time - start_time):.0f}s")

        # TODO: look at model saving below
        """
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        with open(os.path.join(mvae_dir, "pt_losses" + date_str + ".txt"), 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')
        print("Models saved\nFinished")
        """

        if save_models:
            print("Saving Models")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Current timestamp
            final_epoch_loss = epoch_loss  # Replace 'epoch_loss' with the actual value

            # Define model file names
            pt_model_file = f"model_{timestamp}.pt"
            encoder_model_file = f"encoder_model_{timestamp}.pt"
            decoder_model_file = f"decoder_model_{timestamp}.pt"
            epoch_loss_file = f"epoch_loss_{final_epoch_loss}.txt"  # Optionally include epoch loss in the file name

            encoder_path = os.path.join(models_dir, encoder_model_file)
            torch.save(encoder.state_dict(), encoder_path)
            print(f"Saved {encoder_path}")

            decoder_path = os.path.join(models_dir, decoder_model_file)
            torch.save(pt_decoder.state_dict(), decoder_path)
            print(f"Saved {decoder_path}")

        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        with open("pt_losses" + date_str + ".txt", 'w') as f:
            # TODO: maybe remove this?
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            for i, loss in enumerate(losses):
                f.write(f'{i + 1}  {loss}\n')

        plt.plot(losses)
        plt.title("Pre-trainer losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        plt.savefig('pt_losses' + date_str + '.png')
        plt.show()
        plt.close()

    if check_masking_and_infilling:
        vae_model.eval()
        for _ in range(4):
            patch_masker.test(vae_model, pt_dataloader, True, device)

    print("MVAE Script complete")

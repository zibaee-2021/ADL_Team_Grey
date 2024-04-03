# External packages
import torch
import torch.nn as nn
import time
import random
import numpy as np

# our code
from src.utils.paths import *
from src.utils.device import get_optimal_device
from src.finetuning.data_handler import (
    view_batch,
    view_training,
    OxfordPetDataset
)
from src.finetuning.networks_pt import (
    SemanticSegmenter # ,SegmentationClassifier ## unused
)
from src.shared_network_architectures.networks_pt import (
    VisionTransformerEncoder,
    VisionTransformerDecoder
)

## Control
## Training
check_oxford_batch = True
run_semantic_training = True
check_semantic_segmentation = True
save_models = True
load_models = False

## Testing
# check_oxford_batch = True
# run_semantic_training = False
# check_semantic_segmentation = True
# save_models = False
# load_models = True

## Definitions
params = {
    # # image
    "image_size": 224,  # number of pixels square
    "num_channels": 3,  #  RGB image -> 3 channels       # RGB image -> 3 channels
    "patch_size": 16,

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
    "segmenter_classes": 3,  # image, background, boundary
}


# Train-test split
train_size = 0.8
val_size = 0.1
test_size = 1.0 - train_size - val_size

# Hyper-parameters
ft_batch_size = 8
ft_num_epochs = 1 # 30
ft_lr = 0.1
ft_step = 5
ft_gamma = 0.2

# file paths
data_dir = os.path.join(datasets_dir,"Animals-10/raw-img/")
model_file = "masked_autoencoder_model.pth"
encoder_file = "masked_autoencoder_encoder.pth"
decoder_file = "masked_autoencoder_decoder.pth"
segmentation_model_file = "semantic_segmentation_model.pth"
segmentation_encoder_file = "segmentation_encoder.pth"
segmentation_decoder_file = "segmentation_decoder.pth"
oxford_path = oxford_3_dir
oxford_classes = (
    'Abyssinian_cat', 'american_bulldog_dog', 'american_pit_bull_terrier_dog', 'basset_hound_dog',
    'beagle_dog', 'Bengal_cat', 'Birman_cat', 'Bombay_cat', 'boxer_dog', 'British_Shorthair_cat',
    'chihuahua_dog', 'Egyptian_Mau_cat', 'english_cocker_spaniel_dog', 'english_setter_dog',
    'german_shorthaired_dog', 'great_pyrenees_dog', 'havanese_dog', 'japanese_chin_dog', 'keeshond_dog',
    'leonberger_dog', 'Maine_Coon_cat', 'miniature_pinscher_dog', 'newfoundland_dog', 'Persian_cat',
    'pomeranian_dog', 'pug_dog', 'Ragdoll_cat', 'Russian_Blue_cat', 'saint_bernard_dog', 'samoyed_dog',
    'scottish_terrier_dog', 'shiba_inu_dog', 'Siamese_cat', 'Sphynx_cat', 'staffordshire_bull_terrier_dog',
    'wheaten_terrier_dog', 'yorkshire_terrier_dog'
)

ft_num_classes = len(oxford_classes)
report_every = 100

# test image
test_image_path = os.path.join(oxford_3_dir, "images/Abyssinian_1.jpg")

if __name__ == '__main__':

    # Set seeds for random number generator
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Pre-trained models
    model_path = os.path.join(models_dir, model_file)
    encoder_path = os.path.join(models_dir, encoder_file)
    decoder_path = os.path.join(models_dir, decoder_file)
    print(f"{data_dir = }\n{model_path = }\n{encoder_path = }\n{decoder_path = }")

    # Semantic Segmentation
    segmentation_model_path = os.path.join(models_dir, segmentation_model_file)
    segmentation_encoder_path = os.path.join(models_dir, segmentation_encoder_file)
    segmentation_decoder_path = os.path.join(models_dir, segmentation_decoder_file)
    print(f"{oxford_path = }\n{segmentation_model_path = }\n{segmentation_encoder_path = }\n"
          f"{segmentation_decoder_path = }\n{test_image_path = }")

    device = get_optimal_device()

    ## Initialize models
    encoder = VisionTransformerEncoder(params)
    segmentation_decoder = VisionTransformerDecoder(params)
    ft_segmentation_model = SemanticSegmenter(encoder, segmentation_decoder).to(device)

    if load_models and os.path.isfile(segmentation_decoder_path):
        print("Loading pre-saved segmentation model")
        segmentation_decoder.load_state_dict(torch.load(segmentation_decoder_path), strict=False)
        encoder.load_state_dict(torch.load(encoder_path), strict=False)
        ft_segmentation_model.load_state_dict(torch.load(segmentation_model_path), strict=False)

    # load data
    oxford_dataset = OxfordPetDataset(image_dir=os.path.join(oxford_path, "images"),
                                      label_dir=os.path.join(oxford_path, "annotations/trimaps"),
                                      parameters=params)
    train_loader, val_loader, test_loader = oxford_dataset.split_dataset(train_size, val_size, test_size,
                                                                             ft_batch_size)

    ############################
    # train semantic segmentation
    if run_semantic_training:
        print("In semantic segmentation training")  # images need to be in one folder per class
        start_time = time.perf_counter()

        if check_oxford_batch:
            view_batch(train_loader)

        # Define loss function and optimizer
        ft_criterion = nn.CrossEntropyLoss()
        ft_optimizer = torch.optim.Adam([{'params': ft_segmentation_model.encoder.parameters()},
                                         {'params': ft_segmentation_model.decoder.parameters()}],
                                        lr=ft_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(ft_optimizer, step_size=ft_step, gamma=ft_gamma)

        losses = []
        for epoch in range(ft_num_epochs):
            epoch_start_time = time.perf_counter()
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
                if its % report_every == (report_every - 1):  # print every report_every mini-batches
                    curr_time = time.perf_counter() - start_time
                    print('Epoch [%d / %d],  %d image minibatch [%4d / %4d], cumulative running loss: %.4f, uptime: %.2f' % (
                            epoch + 1, ft_num_epochs, ft_batch_size, its + 1, len(train_loader),
                            running_loss / len(train_loader), curr_time))
            scheduler.step()
            epoch_end_time = time.perf_counter()
            losses.append(running_loss / len(train_loader))
            print(
                f"Epoch [{epoch + 1}/{ft_num_epochs}] completed in {(epoch_end_time - epoch_start_time):.0f}s, Loss: {running_loss / len(train_loader):.4f}")
            view_training(ft_segmentation_model, val_loader, device, )
        end_time = time.perf_counter()
        print(f"Segmentation training finished after {(end_time - start_time):.0f}s")

        # save the trained model and losses
        if save_models:
            torch.save(encoder.state_dict(), segmentation_encoder_path)
            torch.save(segmentation_decoder.state_dict(), segmentation_decoder_path)
            torch.save(ft_segmentation_model.state_dict(), segmentation_model_path)
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        with open(os.path.join(fine_tuning_dir, "ft_losses" + date_str + ".txt"), 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')
        print("Fine tune models saved\nFinished")

    ###################################
    # demonstrate semantic segmentation
    if check_semantic_segmentation:
        print("Test semantic segmentation")
        view_training(ft_segmentation_model, test_loader, device)
    print("Semantic segmentation finished")

    print("Fine-tuning script complete")
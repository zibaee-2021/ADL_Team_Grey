import unittest
from unittest import TestCase
import src.MVAE_2903_working as mvae


class MVAETests(TestCase):

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

    def test_mask_tester(self, patch_masker, image_file):


        mvae.mask_tester(patch_masker, image_file)
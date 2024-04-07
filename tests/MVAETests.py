from unittest import TestCase
import src.older_versions.MVAE_2903_working as mvae
import torch
import torch.nn.functional as F
import src.IoUMetric as ioumet


class MVAETests(TestCase):

    parameters = {
        # image
        "image_size": 224,  # number of pixels square
        "num_channels": 3,  # RGB image -> 3 channels
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
    }  # Not currently used.

    # def test_mask_tester(self):
    #
    #
    #     mvae.mask_tester(patch_masker, image_file)

    def test_custom_iou_loss(self):
        # preds has shape (2, 3, 2, 2) for (batch_size, channel, height, width)
        preds = torch.tensor([[[[0.8823, 0.9150], [0.3829, 0.9593]],
                               [[0.3904, 0.6009], [0.2566, 0.7936]],
                               [[0.9408, 0.1332], [0.9346, 0.5936]]],
                              [[[0.8694, 0.5677], [0.7411, 0.4294]],
                               [[0.8854, 0.5739], [0.2666, 0.6274]],
                               [[0.2696, 0.4414], [0.2969, 0.8317]]]], requires_grad=True)
        # ground_truth has shape (2, 1, 2, 2) for (batch_size, channel, height, width)
        ground_truth = torch.tensor([[[[0, 1], [2, 0]],
                                      [[1, 2], [1, 1]],
                                      [[1, 2], [1, 1]]],
                                     [[[1, 2], [1, 1]],
                                      [[1, 2], [1, 1]],
                                      [[1, 2], [1, 1]]]], dtype=torch.long)
        actual = ioumet.IoULoss(preds_are_logits=True)(preds, ground_truth)
        actual = actual.item()
        print(actual)
        expected = 1.0935465097427368
        self.assertEqual(expected, actual)

    def test_custom_iou_metric(self):
        torch.manual_seed(42)
        # preds = torch.rand((2, 3, 2, 2), requires_grad=True) # (batch_size, channel, height, width)
        preds_logits = torch.randn(1, 3, 4, 4)  # 1, 3, 4, 4 are the number of number to generate in each of the 4 dims
        # preds_probs = F.softmax(preds_logits, dim=1)
        ground_truth = torch.tensor([[[0, 1, 2, 1], [1, 2, 0, 2], [2, 0, 1, 0], [0, 2, 1, 2]]], dtype=torch.long)
        print(f'ground_truth.shape={ground_truth.shape}')   # shape (1, 4, 4)
        ground_truth = ground_truth.unsqueeze(0)  # shape (1, 1, 4, 4)
        print(f'ground_truth.shape={ground_truth.shape}')
        # ground_truth = torch.randint(0, 3, (2, 1, 2, 2), dtype=torch.long)
        # actual = ioumet.iou_metric(preds_probs, ground_truth, preds_are_logits=False)
        actual = ioumet.iou_metric(preds_logits, ground_truth, preds_are_logits=True)
        print(f'actual={actual}')
        expected = 0.1706485003232956
        self.assertEqual(expected, actual)

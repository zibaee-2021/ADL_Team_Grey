# GROUP19_COMP0197
from unittest import TestCase
import torch
import src.utils.IoUMetric as ioumet


class MVAETests(TestCase):

    def setUp(self) -> None:
        # preds has shape (2, 3, 2, 2) for (batch_size, channel, height, width)
        # representing a batch of 2 images:
        # First image has ones in first half and zeros in second half
        # Second image has zeros in the first half and ones in the second half
        self.preds = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]],
                               [[1.0, 1.0], [1.0, 1.0]],
                               [[1.0, 1.0], [1.0, 1.0]]],
                               [[[0.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [0.0, 0.0]]]])

        # create another tensor of same shape as preds but move it so no overlap
        self.not_overlapping_gt = torch.zeros_like(self.preds)
        offset_height, offset_width = 2, 2
        self.not_overlapping_gt[1, :, :offset_height, :offset_width] = 1

        self.perfectly_overlapping_gt = self.preds.clone()

    def tearDown(self) -> None:
        self.pred = None
        self.not_overlapping_gt = None
        self.perfectly_overlapping_gt = None

    def test_custom_iou_loss_FOR_PERFECTLY_OVERLAPPING_TENSORS(self):  # IOU *LOSS* SHOULD BE 0.0
        # Create two tensors of shape (2,3,2,2) representing a batch of 2 images:
        # First image has ones in first half and zeros in second half
        # Second image has zeros in the first half and ones in the second half
        ACTUAL_LOSS = ioumet.IoULoss(preds_are_logits=False)(self.preds, self.perfectly_overlapping_gt,
                                                             use_negative_log_loss=False)
        ACTUAL_LOSS = ACTUAL_LOSS.item()
        EXPECTED_LOSS = 0.0
        self.assertEqual(EXPECTED_LOSS, ACTUAL_LOSS)

    def test_custom_iou_loss_FOR_TENSORS_NOT_OVERLAPPING_AT_ALL(self):  # IOU *LOSS* SHOULD BE 1.0
        ACTUAL_LOSS = ioumet.IoULoss(preds_are_logits=False)(self.preds, self.not_overlapping_gt,
                                                             use_negative_log_loss=False)
        ACTUAL_LOSS = ACTUAL_LOSS.item()
        EXPECTED_LOSS = 1.0
        self.assertAlmostEqual(EXPECTED_LOSS, ACTUAL_LOSS, places=4)

    def test_custom_iou_metric_FOR_PERFECTLY_OVERLAPPING_TENSORS(self):  # IOU METRIC SHOULD BE 1.0
        ACTUAL_IOU = ioumet.iou_metric(self.preds, self.perfectly_overlapping_gt, preds_are_logits=False)
        ACTUAL_IOU = ACTUAL_IOU.item()
        EXPECTED_IOU = 1.0
        self.assertEqual(EXPECTED_IOU, ACTUAL_IOU)

    def test_custom_iou_metric_FOR_TENSORS_NOT_OVERLAPPING_AT_ALL(self):  # IOU METRIC SHOULD BE 0.0
        ACTUAL_IOU = ioumet.iou_metric(self.preds, self.not_overlapping_gt, preds_are_logits=False)
        ACTUAL_IOU = ACTUAL_IOU.item()
        EXPECTED_IOU = 0.0
        self.assertAlmostEqual(EXPECTED_IOU, ACTUAL_IOU, places=4)

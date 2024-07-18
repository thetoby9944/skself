from pathlib import Path

import unittest

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"


import segmentation_models as sm
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import iou_score

from skself.partial_annotations.lazy_model import LazySegmentationModel
from skself.data import segmentation_dataset_from_folders

# Define a simple test case
class TestLazySegmentationModel(unittest.TestCase):
    def setUp(self):
        color_dict = {
            0: (0, 0, 255),
            1: (255, 52, 255),
            2: (255, 0, 0),
        }
        n_labelled_classes = len(color_dict) - 1
        unlabelled_class = 2

        self.ds = segmentation_dataset_from_folders(
            Path(__file__).parent / Path("data/example_images"),
            Path(__file__).parent / Path("data/example_masks"),
            color_dict=color_dict,
            verbose=True,  # Enable for printing the dataset
            batch_size=3  # Should not be bigger than max number of images
        )
        print(self.ds)
        # Create a LazyModel instance that outputs 2 channels and ignores the last channel in the labels
        self.lazy_model = LazySegmentationModel(
            sm.Unet('resnet34', input_shape=(256, 256, 3), classes=n_labelled_classes),
            ignore_channel_index=unlabelled_class
        )

        # Compile the LazyModel with a binary cross-entropy loss and accuracy metric
        self.lazy_model.compile(optimizer='adam', loss=DiceLoss(), metrics=['accuracy', iou_score])

    def test_model_fit(self):
        # Test the fit method of the LazyModel runs without errors
        self.lazy_model.fit(self.ds, epochs=1, batch_size=4)

    def test_model_fit_andevaluate(self):
        # Test the evaluate method of the LazyModel
        self.lazy_model.fit(self.ds, epochs=1, batch_size=4)
        loss, accuracy, iou = self.lazy_model.evaluate(self.ds)
        self.assertTrue(loss != None)
        self.assertTrue(accuracy > 0)
        self.assertTrue(iou > 0)


if __name__ == '__main__':
    unittest.main()

from pathlib import Path

from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import iou_score

from skself.partial_annotations.lazy_model import LazyModel

import tensorflow as tf
import segmentation_models as sm
import unittest


# Define a simple test case
class TestLazyModel(unittest.TestCase):
    def setUp(self):

        # Create some random dummy data for testing
        # 32 images 256x256 images with 3 channels between 0 and 1
        self.x_train = tf.random.uniform((32, 256, 256, 3), minval=0, maxval=1)
        # 32 binary segmentation targets for 256x256 images with 3 channels / masks per target
        self.y_train = tf.cast(tf.random.uniform((32, 256, 256, 3), minval=0, maxval=2, dtype=tf.int32),
                               tf.float32)

        # Create a LazyModel instance that outputs 2 channels and ignores the last channel in the labels
        self.lazy_model = LazyModel(
            sm.Unet('resnet34', input_shape=(256, 256, 3), classes=2),
            ignore_channel_index=2
        )

        # Compile the LazyModel with a binary cross-entropy loss and accuracy metric
        self.lazy_model.compile(optimizer='adam', loss=DiceLoss(), metrics=['accuracy', iou_score])

    def test_model_fit(self):
        # Test the fit method of the LazyModel runs without errors
        self.lazy_model.fit(self.x_train, self.y_train, epochs=1, batch_size=4)

    def test_model_evaluate(self):
        # Test the evaluate method of the LazyModel
        loss, accuracy, iou = self.lazy_model.evaluate(self.x_train, self.y_train)
        self.assertTrue(loss != None)
        self.assertTrue(accuracy > 0)
        self.assertTrue(iou > 0)

if __name__ == '__main__':
    unittest.main()
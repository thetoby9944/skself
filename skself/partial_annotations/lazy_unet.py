import segmentation_models as sm
import tensorflow as tf

from typing import Dict, Any

from sklearn.base import TransformerMixin
from scikeras.wrappers import BaseWrapper


class LazyUnet(tf.keras.Model):
    unet: tf.keras.Model

    def __init__(self, unet: tf.keras.Model, ignore_channel=0):
        super(LazyUnet, self).__init__()
        self.unet = unet
        self.ignore_channel = ignore_channel

    def masked_unet(self, unet_output, inputs):
        # Extract the last mask from the inputs
        ignore_mask = inputs[:, :, :, self.ignore_channel]  # Get the channel that should be ignored

        # Element-wise multiply the Unet output with the inverted binary mask
        # to ignore all pixels marked in the ignore_mask
        return unet_output * (1 - ignore_mask)

    def call(self, inputs):
        x = self.unet(inputs)
        y = self.masked_unet(x, inputs)
        return y


class LazyUnet(BaseWrapper, TransformerMixin):
    """A class that enables transform and fit_transform.
    """

    unet: BaseWrapper
    lazy_unet: BaseWrapper
    ignore_channel: int

    def _keras_build_fn(self, ignore_channel: int, meta: Dict[str, Any]):
        self.ignore_channel = ignore_channel
        model = LazyUnet(
            unet = sm.Unet(
                classes = meta["n_classes_"] - 1
            ),
            ignore_channel=ignore_channel
        )
        self.unet = BaseWrapper(model.unet)
        self.lazy_unet = BaseWrapper(model)
        return self.lazy_unet

    def _initialize(self, X, y=None):
        X, y = super()._initialize(X=X, y=y)
        self.unet.initialize(X, y)
        self.lazy_unet.initialize(X, y)
        return X, y

    def initialize(self, X, y):
        self._initialize(X=X, y=y)
        return self

    def fit(self, X, y, sample_weight=None) -> "LazyUnet":
        super().fit(X=X, y=y, sample_weight=sample_weight)
        # at this point, encoder_model_ and decoder_model_
        # are both "fitted" because they share layers w/ model_
        # which is fit in the above call
        return self

    def score(self, X, y) -> float:
        # Note: we use 1-MSE as the score
        # With MSE, "larger is better", but Scikit-Learn
        # always maximizes the score (e.g. in GridSearch)
        y_masked =
        return 1 - sm.metrics.iou_score(self.unet.predict(X), y)

    def transform(self, X) -> np.ndarray:
        X: np.ndarray = self.feature_encoder_.transform(X)
        return self.encoder_model_.predict(X)

    def inverse_transform(self, X_tf: np.ndarray):
        X: np.ndarray = self.decoder_model_.predict(X_tf)
        return self.feature_encoder_.inverse_transform(X)
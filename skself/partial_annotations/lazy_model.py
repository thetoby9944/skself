import tensorflow as tf


def pop_channel(tensor, channel_to_remove):
    # Get the number of channels in the tensor
    num_channels = tf.shape(tensor)[-1]

    # Convert the channel index to a non-negative index
    if channel_to_remove < 0:
        channel_to_remove += num_channels

    # Split the tensor into three parts: before, channel, and after
    before_channel = tensor[..., :channel_to_remove]
    removed_channel = tensor[..., channel_to_remove:channel_to_remove+1]
    after_channel = tensor[..., channel_to_remove + 1:]

    # Concatenate the two parts back together along the channel axis
    result_tensor = tf.concat([before_channel, after_channel], axis=-1)

    return result_tensor, removed_channel



class LazyLossWrapper(tf.keras.losses.Loss):
    def __init__(self, base_loss, mask_index=-1):
        super(LazyLossWrapper, self).__init__()
        self.base_loss = base_loss
        self.mask_index = mask_index

    def call(self, y_true, y_pred):
        # Split the mask from y_true based on the user-specified index
        y_true, mask = pop_channel(y_true, self.mask_index)

        # Apply the mask to y_pred
        masked_y_pred = y_pred * (1 - mask)

        # Calculate the loss using the user-provided loss function
        loss = self.base_loss(y_true, masked_y_pred)

        return loss

class LazyMetricWrapper(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, metric_fn, mask_index=-1, name=None, **kwargs):
        name = name or (metric_fn.name) if hasattr(metric_fn, "name") else metric_fn.__name__
        super(LazyMetricWrapper, self).__init__(metric_fn, name=name, **kwargs)

        self.mask_index = mask_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract the mask from y_true based on the user-specified index
        y_true, mask = pop_channel(y_true, self.mask_index)

        # Apply the mask to y_pred
        masked_y_pred = y_pred * (1 - mask)

        # Call the metric_fn to update the metric value
        super(LazyMetricWrapper, self).update_state(y_true, masked_y_pred, sample_weight)


# Define the custom U-Net wrapper class
class LazyModel(tf.keras.Model):
    def __init__(self, base_model, ignore_channel_index=-1, **kwargs):
        super(LazyModel, self).__init__(**kwargs)
        self.base_unet = base_model
        self.mask_index = ignore_channel_index

    def compile(
            self,
            optimizer="rmsprop",
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            jit_compile=None,
            **kwargs
    ):
        if isinstance(loss, str):
            # Convert the loss string to a loss function
            loss = tf.keras.losses.get(loss)

        if metrics is not None:
            if isinstance(metrics, str):
                # Convert the metrics string to a list of metric functions
                metrics = [tf.keras.metrics.get(metrics)]
            elif isinstance(metrics, list):
                # Convert each metric in the list to a metric function
                metrics = [
                    tf.keras.metrics.get(metric) if isinstance(metric, str) else metric
                    for metric in metrics
                ]

        if loss is not None:
            # Wrap the loss function with the LazyLossWrapper using the specified mask_index
            loss = LazyLossWrapper(loss, mask_index=self.mask_index)

        if metrics is not None:
            # Wrap each metric with the LazyMetricWrapper using the specified mask_index
            metrics = [LazyMetricWrapper(metric, mask_index=self.mask_index) for metric in metrics]

        # Call the compile method of the base_unet with the wrapped loss and metrics
        self.base_unet.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            **kwargs
        )

    def fit(self, *args, **kwargs):
        # Call the call method of the base_unet with the input tensor
        return self.base_unet.fit(*args, **kwargs)

    def call(self, *args, **kwargs):
        # Call the call method of the base_unet with the input tensor
        return self.base_unet.call(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        # Call the call method of the base_unet with the input tensor
        return self.base_unet.evaluate(*args, **kwargs)



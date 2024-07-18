from skself.data.images_from_directory import DatasetBuilder
import albumentations as A
import tensorflow as tf

def segmentation_dataset_from_folders(
        image_folder,
        mask_folder,
        batch_size = 8,
        validation_split=0,
        color_dict=None,
        verbose=False,
        shuffle=True,
        subset=None,
        width = 256,
        height = 256,
        crop_to_aspect_ratio= True,
        seed = 48,
) -> tf.data.Dataset :
    if color_dict is None:
        color_dict = {
            0: (0, 0, 255),
            1: (255, 255, 255),
        }
    return DatasetBuilder(
        pairing_mode="result_only",  # "result_only", "result_with_original"
        create_artificial_anomalies=False,
        validation_split=validation_split,
        color_dict=color_dict,
        shuffle = shuffle,
        peek = verbose,
        image_directory = image_folder,
        mask_directory = mask_folder,
        drop_masks = False,
        subset = subset,
        width = width,
        height = height,
        repeat = False,
        anomaly_size= None,
        process_deviation = None,
        global_transform = None,
        anomaly_composition = A.Compose([]),
        batch_size = batch_size,
        seed = seed,
        crop_to_aspect_ratio = crop_to_aspect_ratio
    ).ds
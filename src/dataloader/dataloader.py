# /content/drive/MyDrive/fake-news-multimodal/src/dataloader/dataloader.py

import tensorflow as tf

def load_dataset(main_dir):
    """Loads image dataset from directory"""
    data = tf.keras.utils.image_dataset_from_directory(main_dir)
    return data

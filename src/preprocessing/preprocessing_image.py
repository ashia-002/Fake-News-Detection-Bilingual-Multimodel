# /content/drive/MyDrive/CNN/src/models/cnn/preprocessing.py

import tensorflow as tf

def normalize_data(data):
    """Normalize pixel values to [0,1]"""
    return data.map(lambda x, y: (x / 255, y))

def split_data(data, train_ratio=0.7, val_ratio=0.2):
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio) + 1
    test_size = total - train_size - val_size

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    return train, val, test

def optimize_pipeline(dataset, buffer_size=1000):
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.shuffle(buffer_size=buffer_size)
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)

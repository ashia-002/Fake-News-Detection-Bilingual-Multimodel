# /content/drive/MyDrive/fake-news-multimodal/src/train/train_cnn.py

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

def train_model(model, train, val, log_dir, save_path):
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(train,
                        epochs=20,
                        validation_data=val,
                        callbacks=[tensorboard_callback, early_stopping])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✅ Model saved to: {save_path}")
    return history

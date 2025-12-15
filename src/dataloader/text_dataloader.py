import os
import torch
from torch.utils.data import TensorDataset
import torch

class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_text_data():
    train_path = "/content/drive/MyDrive/fake-news-multimodal/textdata/train_text_tensors.pt"
    test_path  = "/content/drive/MyDrive/fake-news-multimodal/textdata/test_text_tensors.pt"

    train = torch.load(train_path)
    test  = torch.load(test_path)

    train_dataset = TextTensorDataset(
        {"input_ids": train["input_ids"], "attention_mask": train["attention_mask"]},
        train["labels"]
    )

    test_dataset = TextTensorDataset(
        {"input_ids": test["input_ids"], "attention_mask": test["attention_mask"]},
        test["labels"]
    )

    print(f"Loaded training dataset with {train['labels'].shape[0]} samples.")
    print(f"Loaded testing dataset with {test['labels'].shape[0]} samples.")

    return train_dataset, test_dataset

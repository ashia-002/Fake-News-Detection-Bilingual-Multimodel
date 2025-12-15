import torch
from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, tf_dataset, text_dataset, device='cpu'):
        self.tf_dataset = list(tf_dataset)  # keep dataset as list of (image, label)
        self.text_dataset = text_dataset
        self.device = device

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        img, lbl = self.tf_dataset[idx]
        # Convert single image to tensor and permute channels
        img_tensor = torch.tensor(img.numpy(), dtype=torch.float32).permute(2,0,1).to(self.device)
        lbl_tensor = torch.tensor(lbl.numpy(), dtype=torch.long).to(self.device)
        
        text_item = self.text_dataset[idx]
        input_ids = text_item['input_ids'].to(self.device)
        attention_mask = text_item['attention_mask'].to(self.device)

        return img_tensor, input_ids, attention_mask, lbl_tensor

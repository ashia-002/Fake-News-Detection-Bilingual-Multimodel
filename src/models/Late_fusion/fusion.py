import torch
import torch.nn as nn
import tensorflow as tf
from transformers import XLMRobertaModel

class FusionModel(nn.Module):
    def __init__(self, cnn_model, text_checkpoint, num_classes=2):
        super(FusionModel, self).__init__()
        self.device = device

        # CNN feature extractor (remove final dense layer)
        self.cnn_feature_extractor = tf.keras.Model(
            inputs=cnn_model.input,
            outputs=cnn_model.layers[-3].output  # Dense layer before Dropout and final Dense
        )

        # Load pretrained XLM-R as feature extractor
        self.text_model = XLMRobertaModel.from_pretrained(text_checkpoint)
        self.text_model.eval()  # freeze by default
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Fusion classifier
        self.fc = nn.Sequential(
            nn.Linear(128 + 768, 256),  # 128 CNN features + 768 XLM-R CLS
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, images, input_ids, attention_mask):
        # CNN features
        cnn_features = self.cnn_feature_extractor(images)  # tf.Tensor
        if isinstance(cnn_features, tf.Tensor):
            cnn_features = torch.tensor(cnn_features.numpy(), dtype=torch.float32, device=self.device)

        # XLM-R features (CLS token)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        text_features = text_features.to(self.device)

        # Concatenate features
        fused = torch.cat((cnn_features, text_features), dim=1)
        return self.fc(fused)

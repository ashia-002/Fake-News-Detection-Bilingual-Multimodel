#/content/drive/MyDrive/fake-news-multimodal/src/models/XML_Roberta/XLM_R_Model.py
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer


# Model Configuration
MODEL_NAME = 'xlm-roberta-base'
NUM_LABELS = 2 # Binary classification

def build_xlm_model(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):

    print(f"Building model architecture: {model_name}")
    
    # 1. Load Configuration
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    
    # 2. Load Model weights with the new classification head
    # This automatically handles adding the necessary linear layer on top.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    
    print(f"Model initialized with {num_labels} output classes.")
    return model


def get_tokenizer(model_name: str = MODEL_NAME):
    """Loads the necessary tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_name)
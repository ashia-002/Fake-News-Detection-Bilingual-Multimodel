#/content/drive/MyDrive/fake-news-multimodal/src/evaluate/evaluate_roberta.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_text_metrics(p):
    """
    Computes key classification metrics (Accuracy, F1, Precision, Recall).
    
    Args:
        p (EvalPrediction): A tuple containing predictions (logits) and true labels.
        
    Returns:
        dict: Dictionary of metrics.
    """
    preds = np.argmax(p.predictions, axis=1)
    
    # Calculate main classification metrics (using 'binary' average)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, 
        average='binary',
        zero_division=0
    )
    acc = accuracy_score(p.label_ids, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# /content/drive/MyDrive/fake-news-multimodal/src/evaluate/evaluate_cnn.py

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

def evaluate_model(model, test):
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    y_true, y_pred = [], []
    for batch in test.as_numpy_iterator():
        x, y = batch
        yhat = model.predict(x)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)

        yhat_bin = np.round(yhat).astype(int)
        y_true.extend(y)
        y_pred.extend(yhat_bin)

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- Evaluation Metrics ---")
    print(f"Precision: {pre.result().numpy():.4f}")
    print(f"Recall: {re.result().numpy():.4f}")
    print(f"Accuracy: {acc.result().numpy():.4f}")
    print(f"F1 Score: {f1:.4f}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

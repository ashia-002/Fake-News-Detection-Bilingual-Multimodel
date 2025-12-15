# /content/drive/MyDrive/fake-news-multimodal/src/train/train_roberta.py

import os
import torch
from transformers import TrainingArguments, Trainer
from models.XML_Roberta.XLM_R_Model import get_tokenizer, MODEL_NAME
from evaluate.evaluate_roberta import compute_text_metrics

OUTPUT_DIR_RUN = "/content/drive/MyDrive/fake-news-multimodal/models/xlm_roberta"
LOGGING_DIR    = "/content/drive/MyDrive/fake-news-multimodal/logs/xlm_roberta"

def train_text_model(model, train_dataset, eval_dataset):

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_RUN,

        # Training settings
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-5,

        # Logging
        logging_dir=LOGGING_DIR,
        logging_steps=50,
        report_to="none",

        # Keep last checkpoint only
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_text_metrics,
        tokenizer=get_tokenizer(MODEL_NAME),
    )

    print("\n--- Starting XLM-RoBERTa Fine-Tuning ---")
    trainer.train()

    # Evaluate on test set to find best checkpoint
    print("\n--- Running Final Evaluation on Test Set ---")
    evaluation_results = trainer.evaluate(eval_dataset)

    # Save model manually as a fusion-ready single file
    os.makedirs(OUTPUT_DIR_RUN, exist_ok=True)
    fusion_path = os.path.join(OUTPUT_DIR_RUN, "text_encoder_fusion.pt")
    torch.save(trainer.model.state_dict(), fusion_path)
    print(f"\nFusion-ready BEST MODEL saved to: {fusion_path}")

    # Optional: save in HuggingFace format for reuse
    hf_path = os.path.join(OUTPUT_DIR_RUN, "hf_format")
    os.makedirs(hf_path, exist_ok=True)
    trainer.save_model(hf_path)
    print(f"📁 HuggingFace BEST MODEL saved to: {hf_path}")

    return trainer.model, evaluation_results

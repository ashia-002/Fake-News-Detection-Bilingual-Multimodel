# Fake News Detection Bilingual Multimodel

## Overview

This project is a **multimodal bilingual (English + Bangla) fake news detection system** designed to analyze social media images containing both text and visuals (from platforms like Facebook, YouTube, Instagram). The goal is to determine whether the news content in the image is **fake or real** by jointly analyzing both image and extracted text.

---

## Problem Statement

Fake news classification is based on combined cues from both the image and text:

| Image Status | Text Status | Final Label |
| ------------ | ----------- | ----------- |
| Fake         | Fake        | Fake        |
| Real         | Fake        | Fake        |
| Fake         | Real        | Fake        |
| Real         | Real        | Real        |

---

## Model Architecture

The system integrates two modalities — image and text — using a fusion approach:
```
Input (Image)
|
|----> Custom CNN / ResNet50 -----> Classification (Image)
|
|----> Text Extraction -----> mBERT / XML-RoBERTa -----> Classification (Text)
|
|-------------------------------------------------------> Fusion
|
Real / Fake

             ┌─────────────────┐
             │   Input Image   │
             └─────────────────┘
                      │
       ┌──────────────┴──────────────┐
       │                             │
┌──────────────────────┐ ┌────────────────────────┐
│ Custom CNN /         │ │ Text Extraction        │
│ ResNet50             │ │ (OCR + Preprocess)     │
└──────────────────────┘ └────────────────────────┘
          │                             │
┌──────────────────────┐ ┌────────────────────────┐
│ Image Classification │ │ Text Classification    │
└──────────────────────┘ │ mBERT / XML-RoBERTa    │
                         └────────────────────────┘
          │                             │
          └──────────────┬──────────────┘
                         │
               ┌─────────────────┐
               │ Fusion          │
               │ Combine image & │
               │ text predictions│
               └─────────────────┘
                        │
               ┌─────────────────┐
               │ Real / Fake     │
               └─────────────────┘

```

---

## Folder Structure

* **CNN_Pipeline/**  
  Contains the full global CNN pipeline implementation for image classification.

* **LargeDataset/**  
  Contains 9,000+ images used for training and evaluation.

* **data/**  
  Contains approximately 4,000 images, separated into real and fake classes.

* **logs/**  
  Contains detailed training logs for the different individual models.

* **models/**  
  Holds saved trained model files (e.g., CNN weights, transformers, xml-RoBERTa trained weights, fusion model).

* **notebooks/**  
  Jupyter notebooks for training and evaluating each model.

* **src/**  
  Currently contains custom CNN model, XML-RoBERTa model and Fusion model implementation and related code.
  
This architecture includes the following components in the `src/` directory:

   - **dataloader/** — Scripts for loading and preprocessing image data.
   - **evaluate/** — Code for evaluating model predictions and performance metrics.
   - **models/** — Model architectures and model definitions used in image classification.
   - **preprocessing/** — Image and text preprocessing routines to prepare data for modeling.
   - **train/** — Training loops and utility functions to train and optimize models.


---

## Usage

* Train or evaluate models using the scripts and notebooks inside `notebooks/` and `src/`.
* Use the pretrained models in `models/` for inference.
* Check training progress and debugging info in `logs/`.

---

If you want to contribute or improve the models, please follow the existing structure and document your changes.

---

**Author:** ashia-002 & farhanaz-08

**Contact:** ashia.sultana.maisha@gmail.com  

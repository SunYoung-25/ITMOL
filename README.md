# ITMol
##### A Multimodal Foundation Model for Molecular Property Prediction via Image-Text Pre-training
  - [Overview](#overview)
  - [Introduction](#Introduction)
  - [Dataset](#Dataset)
  - [Usage](#Usage)
  - [Citation](#citation)
## Overview

![ITMol Architecture](assets/overview.jpg)

The overview of the proposed ITMol model. (a) The data construction phase results in a dataset of 500k molecular image-text pairs.
(b) The pre-training phase aligns text and image modalities. (c) The fine-tuning phase predicts molecular properties.

## Introduction

In this work, we present a molecular image-text foundation model, named ITMol, pretrained on 500k molecular image-text pairs. ITMol effectively designs three self-supervised learning strategies on molecular image-text pairs and adopts cross-attention mechanism to capture molecular representation.

## Dataset

The ITMol dataset is a multimodal dataset designed for molecular property prediction. It includes molecular images and text descriptions that are preprocessed for training.

### **Download Pretraining Dataset**
You can download the pretraining dataset from the following link:

[Pretraining Dataset - Baidu Pan](https://pan.baidu.com/s/1bEOSEe8q5EKIBEOyUPl4HA?pwd=7pdi) (Password: 7pdi)

### **Data Preprocessing**
1. **SMILES to Images**:
   - Use RDKit to convert the `SMILES` column from the dataset into molecular structure images.
   - Save the generated images in the `/images` directory.
   
2. **Text Descriptions**:
   - Use the `Description` column as the text data for the molecules.
   
3. **Processing Script**:
   - Use `dataset.py` to process the dataset and organize it into the required format for training.

### **Steps to Process the Dataset**
Run the following script to preprocess the dataset:

```bash
python dataset.py
```

## Usage

To train and fine-tune the ITMol model, follow these steps:

### Pretraining
Pretrain the ITMol model using the preprocessed dataset with the following command:

```bash
python pretrain.py
```
### Fine-Tuning
Fine-tune the pretrained ITMol model on downstream tasks, such as molecular property prediction, using the following command:

```bash
python pretrain.py
```

## Citation

# Unsupervised-Anomaly-Detection-on-MVTecAD - This repo is forked from Shiva Arabi
This repository implements an unsupervised anomaly detection pipeline inspired by PatchCore using the MVTec AD (Tile class) dataset in google colab environment. The goal is to detect and localize surface-level defects (e.g., cracks, glue strips, oil, roughness) in industrial tile images using pre-trained deep features.

Link to dataset: https://www.kaggle.com/datasets/ipythonx/mvtec-ad

Project Highlights:
Dataset: MVTec AD – tile category with labeled training and testing images (defects include crack, glue_strip, gray_stroke, oil, rough, and good).
Backbone Model: Pretrained ResNet-50 used as a fixed feature extractor, with hooks to extract intermediate feature maps.
Feature Encoding: Multi-layer deep features from ResNet are pooled and concatenated to build rich patch-level embeddings.
Memory Bank Construction: Feature vectors from "good" training samples are stored as reference for comparison.
Anomaly Scoring: At test time, patch-wise distances to the memory bank are computed. The maximum nearest neighbor distance is used as the anomaly score.
Thresholding: A dynamic threshold is computed from training scores using mean + 3 × std for robust detection.

Evaluation:
AUC-ROC and F1 score are calculated across all test classes.
Confusion matrix and ROC curve are plotted.

Heatmap Visualization:
Per-pixel anomaly heatmaps are generated and upsampled to match input resolution.
A custom function visualizes image, anomaly map, and binary segmentation mask for multiple test images.

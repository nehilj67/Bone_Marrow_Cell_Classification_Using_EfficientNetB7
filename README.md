# Bone Marrow Cell Classification using Deep Learning
Overview

This project focuses on automated classification of bone marrow cell images using deep learning techniques. It leverages a convolutional neural network (CNN) architecture, EfficientNetB7, to accurately classify bone marrow cells from microscopic images, aiming to support medical image analysis and diagnostic workflows.

Problem Statement

Manual bone marrow cell classification is time-consuming and prone to inter-observer variability. This project addresses these challenges by applying deep learning to improve accuracy, consistency, and efficiency in bone marrow cell classification.

Key Features

Deep learning–based medical image classification

EfficientNetB7 architecture for high accuracy and efficiency

Image preprocessing and data augmentation

Performance evaluation using standard classification metrics

Confusion matrix–based analysis

Model Architecture

The project uses EfficientNetB7, a state-of-the-art CNN known for its compound scaling of depth, width, and resolution. This allows the model to capture fine-grained features in bone marrow cell images while remaining computationally efficient.

Dataset

Microscopic images of bone marrow cells

Multiple cell types with labeled ground truth

Preprocessing steps include resizing, normalization, and augmentation (rotation, flipping, zooming)

Note: Dataset is not included in this repository due to size and licensing constraints.

Methodology

Data collection and preprocessing

Train–validation–test split

Model training using EfficientNetB7

Model evaluation using accuracy, precision, recall, and F1-score

Performance analysis using confusion matrices

Results

Achieved high classification accuracy (~95%) using EfficientNetB7

EfficientNetB7 demonstrated strong generalization on unseen data

Effective at capturing subtle morphological differences in bone marrow cells

Model Comparison: EfficientNetB7 vs VGG16

A comparative analysis was conducted between EfficientNetB7 and VGG16 for bone marrow cell classification. VGG16 showed very low accuracy (~8%), indicating poor generalization and limited ability to capture complex cellular patterns. In contrast, EfficientNetB7 achieved ~95% accuracy, benefiting from compound scaling of depth, width, and resolution. This highlights EfficientNetB7 as a significantly more effective and reliable architecture for medical image classification tasks.

Technologies Used

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib / Seaborn




Use Cases

Medical image analysis

Hematological research

Decision support systems in diagnostics

Limitations & Future Work

Validate on larger and more diverse datasets

Extend to multi-class and imbalanced datasets

Deploy as a web or clinical decision-support application

References

Mundekar, R. Bone Marrow Classification Using Deep Learning. Open-source GitHub repository.

Mishra, N., Jahan, I., Nadeem, M. R., Sharma, V. A Comparative Study of ResNet50, EfficientNetB7, InceptionV3, and VGG16 Models.

Guo, L. et al. A classification method to classify bone marrow cells with class imbalance problem. Biomedical Signal Processing and Control.

Chandradevan, R. et al. Machine-based detection and classification for bone marrow aspirate differential counts.

Wang, C.W. et al. Deep learning for bone marrow cell detection and classification on whole-slide images.

Author

Nehil Kishor Joshi

If you find this project useful, feel free to ⭐ the repository.

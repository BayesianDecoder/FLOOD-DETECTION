# Flood detection from satellite images



## Overview
This project utilizes deep learning techniques to detect flood-prone areas from satellite imagery. By leveraging the VGG16 convolutional neural network (CNN) pre-trained on ImageNet, we aim to perform accurate and efficient flood detection. The model has been fine-tuned and tested to ensure high accuracy in identifying water-covered areas during flood events.


## Features

Pre-trained VGG16 CNN: Uses layers from the VGG16 model to extract features from satellite images.

	A. Fine-tuning: Adjusts deeper layers of the model to specialize in detecting features relevant to flood conditions.
 
	B. Global Average Pooling: Reduces the dimensionality of feature maps to simplify the model while retaining essential spatial hierarchies.
 
	C. Binary Classification: Predicts whether a given satellite image segment contains flood areas.



## Model Architecture

The model architecture comprises several key components:
	• Base Model (VGG16): Configured to operate without the top layer to serve as a feature extractor.
 
	• Global Average Pooling Layer: Condenses feature maps to a single 512-length vector per image.
 
	• Prediction Layer: A dense layer with one output unit, applying a sigmoid activation function to predict flood presence.

## Fine-Tuning Details

• Initial Training: The top layers are trained with a base learning rate of 0.0001.

• Unfreezing Layers: Post-initial training, layers from the 15th onwards are unfrozen to allow more specific feature learning.

• Reduced Learning Rate: The learning rate is reduced by a factor of 75 during fine-tuning to make subtle adjustments, preventing overfitting.
 
## Results

• Validation Accuracy: Achieved 98.06% accuracy on the validation set, indicating high generalizability.

• Test Accuracy: Achieved 97.29% on a balanced test dataset, confirming the model's effectiveness in flood detection.

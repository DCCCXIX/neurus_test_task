## Cats and dogs classification and detection task

### Context

This repository contains a notebook with a solution for a detection/classification task.
The task is solved by fine-tuning an EfficientNet_B3 for classification and detection.
EfficientNet, originally created for casual classification tasks, is fine-tuned by adding
several fc layers on top of the original model, splitting the model's output into two
of size 4 and N_CLASS for bounding box coordinates and classes respectively.
MSE loss is optimized for bounding box coordinates and cross entropy is used for classification.

No multi-GPU support or gradient checkpointing is implemented.

### Setup

In order to run this notebook, you'll need to:
 - Install pytorch following instructions in the link below:

https://pytorch.org/get-started/locally/

In order to use GPU, make sure that your pytorch and cuda versions are compatible.
Using GPU is adviced.

To set up prerequisites do in bash:
```bash
pip install requirements.txt
```
macos users (additional step):
```bash
brew install libomp
```

### Dataset

The provided dataset contains 3385 images which belong to one of two categories:
1037 images of cats and 2348 imgages of dogs. Images are supplied with .txt files
containing images' target classes and regions of interest.
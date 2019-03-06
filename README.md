# image_classifier
Jupyter notebook using deep learning to predict terrain classification in Kaggle Amazon competition

Kaggle competition here: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space  
Given satellite imagery of the Amazon, task was to predict which terrain categories (primary rainforest, agriculture etc) were present in each image. Competition was originally run mid-2017.

I used this notebook to take my own stab at the problem. The approach I used gets a mean f2 score of **0.931** on the test set, compared to the competition winning score at the time of **0.933** This approach takes around 5 hours to train on a single GPU.
I've also included a few tweaks I tried to try and improve the score (that didn't work on this dataset, but have been used with success in other competitions).

Overview of solution:

* Stack: Pytorch and fastai libraries running on Quadro M4000 GPU

* Task: multi-label classification on class-imbalanced dataset

* Architecture: pretrained densenet-169 model backbone for transfer learning, custom classification head with adaptive max/meanpooling layer

* Loss Func: binary cross-entropy

* Evaluation metric: mean f2 score (same as micro f2)

* Optim: AdamW

* Data transforms: Random affine dihedrals, minor zoom-and-crop, lighting shifts

* Learning rate schedule: cyclic learning rate schedule with peak lr set by exponential lr search in each phase. Training took place in 4 phases:
    1. Train model head weights seperately; downsampled (128x128) images
    2. Fine tune all model weights; downsampled (128x128) images
    3. Train model head weights seperately; larger (224x224) images
    4. Fine tune all model weights; larger (224x224) images

* Prediction: Prediction probabilities with test-time-augmentation (80% weighting of untransformed image) with prediction threshold for each category of 0.2. Optimal threshold and tta weighting found via parameter search using validation set. 

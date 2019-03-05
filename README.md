# image_classifier
Jupyter notebook using deep learning to predict terrain classification in Kaggle Amazon competition

task: multi-label classification on class-imbalanced dataset

architecture: pretrained densenet base

loss: binary cross-entropy

eval metric: micro f2 score (proxy for competition metric of mean f2 score)

optim: AdamW

learning rate schedule: cyclic learning rate schedule with peak lr set by exponential lr search 

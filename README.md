# MvESR
This is an implementation of Learning Enhanced Specific Representations for Multi-view Feature Learning (MvESR) in Pytorch.
## Requirements
  * Python=3.6.5  
  * Pytorch=1.6.0  
  * Torchvision=0.7.0
## Datasets
The model is trained on AWA/Caltech101/Caltech20/Reuters/CIFAR-10/Flowers-102 dataset, where each dataset are splited into two parts: 60% samples for training, and the rest samples for testing.  We utilize the classification accuracy to evaluate the performance of all the methods.
## Implementation
``
#Train/Test the model on Caltech20 dataset
python MvNNcor_Derive.py --dataset_dir=./mvdata/Caltech101-20 --data_name=Caltech20 --num_classes=20 --num_view=6

``


In this repo, I represent the code base to learn a binary classifier capable of distinguishing between images of fields
and roads. 

The available dataset ([download from here](https://drive.google.com/file/d/1pOKhKzIs6-oXv3SlKrzs0ItHI34adJsT/view)) 
comprises 45 Field images, 108 Road images, and 10 unannotated images designated for testing the trained model. The 10 unannotated images
(consisting of 4 Field images and 6 Road images) can be manually annotated without difficulty.

One significant challenge I faced is the limited size of the proposed dataset, which may
not be sufficient for training state-of-the-art classifiers with convolutional layers (such as
ResNet, VGG, EfficientNet, etc.). Additionally, there is a data imbalance issue that needs
to be addressed to prevent the model from becoming biased toward the majority class.

**Code details** 

1. [split_data.py](./split_data.py) generates csv files of the training, validation and testing sets.

2. [main.py](./main.py) is the main code for training the model.

   [model.py](utils/model.py) contains the applied model architectures.
   [data_loader.py](utils/data_loader.py) is the data_loader code.

3. [evaluation.py](./evaluation.py) is a script for evaluating the learned models.
The selected model can be downloaded from this link [Selected_model](https://drive.google.com/file/d/1E2_sviZQkeGrAujvis4hUh9Lk_Nfl8OI/view?usp=drive_link).

[Report](./Report.pdf) provides the experimental details.

The specified packages and versions used to run this repository codes are listed in [requirements.yml](./requirements.yml). 
To create Conda environment and export requirements, you can use the following command:
```
conda env create -f requirements.yml
```

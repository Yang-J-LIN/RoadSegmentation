# RoadSegmentation

This is the repository for the [Road Segmentation Challenge on AICrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

We used a U Net (adapted from [Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets)) as our predictor, adding downsampling and Sigmoid to the last layers.

The codes are organized as:
* `dataset.py` contains the `RoadSegmentationDataset` class and data augmentation.
* `main.py` contains the code for the training.
* `unet.py` contains the realization of the UNet.
* `visualize.ipynb` provides some code for the visualization of the result and batch generation of outputs.
* `mask_to_submission.py` turns masks into the submission file.

## Installation
1. Clone to the repository to local:

`
git clone https://github.com/Yang-J-LIN/RoadSegmentation.git 
`

2. Install the required packages:

`
conda install --name RoadSegmentation --file requirements.txt
`
3. Download the required dataset. The training set and test set are available on [Road Segmentation Challenge on AICrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation). DeepGlobe dataset is used as external dataset. The DeepGlobe road images are selected and cropped, which can be downloaded from https://drive.google.com/file/d/187NBB0GzuoQ8bAaFdZuaxebvdnvQD4-J/view?usp=sharing.

## Running the code

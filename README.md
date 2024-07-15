## Anomaly Detection
This project include anomaly detection based on Deep Learning.

## Domain
In this case the usuasl data is about dishes with proper serving and the anomaly data is about dishes with uncorrect serving.

## Method Overview 
The main idea is to use a thresholded difference beetwen the reconstruction error of a training image and an anomaly image. The reconstrucion error in the training image should be lower than in the anomaly image.

## Train process
I have train two CNN autoencoder on DSM-50 and DSM-100. Model is compiled and trained with **Adam optimizer**, **MSE loss**, and a **batch size** of 64 for **200 epochs**.

## Datasets
The proposed models have been trained on two datasets: DSM-50 and DSM-100.

## Model
There is a total of 1 models based on the Convolutional Auto-Encoder (CAE) architecture implemented in this project:

* Inception CAE by  https://github.com/natasasdj/anomalyDetection 
  
An autoencoder is a special type of neural network that is trained to copy its input to its output. An autoencoder first encodes the image into a lower dimensional latent representation, then decodes the latent representation back to an image. An autoencoder learns to compress the data while minimizing the reconstruction error.

## Prerequisites

### Dependencies
The main libraries used in this project:
*  `tensorflow == 2.17.0` 
* `keras == 2.4.3`
  
You can see others in `requirements.txt`

### Installation

* Update pip: `pip install -U pip`
* And use: `pip install <library_name>`

### Download 

Download datasets [here](https://disk.yandex.ru/d/5RngW1_VZEflnw) and then move extracted images files to data folder.
Download model weights [here](https://disk.yandex.ru/client/disk/DSM-weights)

## Project Organization
```
├── src                       <- folder containing all mvtec classes.
│   ├── data                  <- folder containing the datasets
|   ├── models                <- folder containing model classes
|   ├── scripts               <- folder containing additional functions
|   ├── trained-models        <- folder containing all trained models
|   ├── example.ipynb         <- demonstrate models results file 
|   ├── train.ipynb           <- fit model file 
├── README.md                 <- readme file.
└── requirements.txt          <- requirement text file containing used libraries.
```

## Authors
* [Vologin Nikita Sergeevich](https://github.com/NikitaVologin)

## References
I have used those github repositories:
* [MVTec-Anomaly-Detection](https://github.com/AdneneBoumessouer/MVTec-Anomaly-Detection/tree/master)
* [anomalyDetection](https://github.com/natasasdj/anomalyDetection/tree/master)
* [2D-and-3D-Deep-Autoencoder](https://github.com/laurahanu/2D-and-3D-Deep-Autoencoder)
* [Image-reconstruction-and-Anomaly-detection](https://github.com/sohamk10/Image-reconstruction-and-Anomaly-detection)

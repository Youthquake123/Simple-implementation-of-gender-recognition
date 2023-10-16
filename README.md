# Simple Implementation of Gender Recognition

This project aims to train a model to recognize gender through images.

## Data Collection

1. Download the image dataset from [this link](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz). The dataset contains the names and pictures of over 5000 individuals, classified into folders.

## Data Processing

2. Based on the information provided in the file `lfw-deepfunneled-gender.txt`, select 4000 distinct male images and 1200 female images from the downloaded pictures.

3. Use 3500 selected male images and 1000 female images as the training set, and the remaining 500 male images and 200 female images as the test set. Each image is cropped to a 200x200-pixel center and resized to 100x100 to reduce memory consumption.

## Model Training

4. Utilize PyTorch to train a model capable of recognizing the gender of individuals from the provided images. Various approaches can be explored to achieve the desired results.

## Result Visualization

5. Implement matplotlib to display the predicted image results. This can involve displaying one or more images as required by the application.

Feel free to contribute and provide feedback for further improvements.







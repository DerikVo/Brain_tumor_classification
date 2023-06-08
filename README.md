# Brain_tumor_classification
This project was a part of my capstone project for General assembly. The goal was to use a neural network to classify brain tumors and have a web app to predict the probability of an image having a brain tumor.

## Background:

According to [Munir, S. Khan, Hanif, and M. Khan (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7794124/) found that Giloma tumors are the most common type of tumor. They found that identification of these tumors has an accuracy of 87% (N = 154), using histopathology also known as "the study of diseased cells and tissues using a microscope" [cancer.gov](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/histopathology). [Morgan (2022)](https://www.cancer.gov/rare-brain-spine-tumor/blog/2022/neuroradiology) explains the first MRI image is typically a baseline and subsequent scans are used to determine if there is a change. Morgan argues that determining the results of an MRI scan can often be an inefficient process due to spending "15 mintues or more arguing about weather something had change on the MRI". [Zacharaki et al. (2009)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2863141/) argues the difficulty of an MRI diagnoses is due to tissue often being heterogeneous and often difficult to distinguish with the human eye.


## Problem Statement:
Because of the difficulty of manual classification of a brain tumor building a classification model would greatly reduce the time spent in the MRI room and help guide the MRI process.

## EDA


## Modeling


## Conclusion


## Recommendations

## Considerations

[Morgan (2022)](https://www.cancer.gov/rare-brain-spine-tumor/blog/2022/neuroradiology) brings up the argument of variation in MRI scans depending on the machine used. Because we don't know what machine was used to get these MRI scans, we cant account for variations in these images. The most significant difference is the resolution of images between machines, which will affect how well our model learns details. This fact is expended on by [Zacharki et al. (2009)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2863141/) which explains brain scans are often difficult to distinguish between the tissues of the brain. So in essence the quality of the machine has varying levels of image clarity which would influence our models performance. One area to EDA area to explore is to see the average pixel density of the images or identify what machine was used for the MRI scans.
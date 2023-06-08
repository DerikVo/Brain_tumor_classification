# Brain_tumor_classification
This project was a part of my capstone project for General assembly. The goal was to use a neural network to classify brain tumors and have a web app to predict the probability of an image having a brain tumor.

## Background:

According to [Munir, S. Khan, Hanif, and M. Khan (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7794124/) found that Giloma tumors are the most common type of tumor. They found that identification of these tumors has an accuracy of 87% (N = 154), using histopathology also known as "the study of diseased cells and tissues using a microscope" [cancer.gov](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/histopathology). As a note, the radiologists reviewing the images had a minimum of five years of post fellowship experience. [Morgan (2022)](https://www.cancer.gov/rare-brain-spine-tumor/blog/2022/neuroradiology) explains the first MRI image is typically a baseline and subsequent scans are used to determine if there is a change. Morgan argues that determining the results of an MRI scan can often be an inefficient process due to spending "15 mintues or more arguing about weather something had change on the MRI". [Zacharaki et al. (2009)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2863141/) argues the difficulty of an MRI diagnoses is due to tissue often being heterogeneous and often difficult to distinguish with the human eye.

[According to AppliedRadiology (2023)](https://appliedradiology.com/articles/the-radiologist-s-gerbil-wheel-interpreting-images-every-3-4-seconds-eight-hours-a-day-at-mayo-clinic) found that radiologist have about three to four seconds to review an image to meet workplace demands. Because of this work demand there is a need to support radiologist in interpreting an MRI scan.

In addition to the radiologist there are a few more stakeholders we need to be aware of.
Stake holders:

|Stakeholder|reason|
|-----------|-------|
|Data Scientist|Develop and train the model|
|Medical Staff|provide domain expertise, validate output, and interpret results. Works with patients|
|Data Team|Collects, cleans, and provides the right data|
|Legal Team|Ensure we are following legal procedure. Ensure we are maintaining confidentiality|
|Regulatory bodies|Ensure we are in compliance with regulation requirements|
|Administrators|Decide how and if we implement the model|
|IT professionals|Ensure the model is functioning smoothly, and first point of contact for trouble shooting|
|Patients|Consent to their data being used. Diagnosis impacts their leaves, so being mis-diagnosed or un-diagnosed has a significant impact|
|MRI manufacturers|May design their MRI based on the needs of our model, or provide insights to improve our model|
|Researchers|Can be a point of collaboration to improve model performance or provide expertise|

## Problem Statement:
Because of the difficulty of manual classification of a brain tumor and the time constraints put on radiologist, building a classification model would greatly reduce the workload of a radiologist. This model will help guide the MRI process by classifying the images so the radiologist will have a reference point when evaluating an MRI scan. This model should only be used as an assistant and not for diagnosis.
For the purposes of this model, we want to limit our false negatives. We want to avoid the model predicting no tumor when in reality there is a tumor. A false positive,predicting there is a tumor when there is no tumor, would simply require a follow up from a radiologist or specialized professional to confirm a diagnosis. In this case the model cares more about the precision and accuracy score of the model.

## EDA


## Modeling


## Conclusion


## Recommendations

## Considerations

[Morgan (2022)](https://www.cancer.gov/rare-brain-spine-tumor/blog/2022/neuroradiology) brings up the argument of variation in MRI scans depending on the machine used. Because we don't know what machine was used to get these MRI scans, we cant account for variations in these images. The most significant difference is the resolution of images between machines, which will affect how well our model learns details. This fact is expended on by [Zacharki et al. (2009)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2863141/) which explains brain scans are often difficult to distinguish between the tissues of the brain. So in essence the quality of the machine has varying levels of image clarity which would influence our models performance. One area to EDA area to explore is to see the average pixel density of the images or identify what machine was used for the MRI scans.
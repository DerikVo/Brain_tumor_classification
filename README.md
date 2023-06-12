# Brain_tumor_classification
This project was a part of my capstone project for General assembly. The goal was to use a neural network to classify brain tumors and have a web app to predict the probability of an image having a brain tumor.

Table of contents
|Section|
|-------|
|[Backgound](#Background)|
|[Problem_Statement](#Problem_Statement)|
|[EDA](#EDA)|
|[Modeling_and_Evaluations](#Modeling_and_Evaluations)|
|[Conclusion](#Conclusion)|
|[Recommendations](#Recommendations)|
|[Limitations](#Limitations)|



## Background:

MRI or magnetic resonance imaging, is a tool used to diagnosis brain tumors. [Pejrimovsky et al. (2022)](https://www.nature.com/articles/s41597-022-01157-0) suggest that are over 150 different types of brain tumors defined by the world health organization. They suggest brain tumors account for a large fraction of potential life loss compared to tumors located on other sites. They suggest these brain tumors have a significant negative impact on an individual’s quality of life. 
Within our dataset we only trained on 4 classes: No tumor, Meningioma, Glioma, and pituitary tumors. [Yildirim, Cengil, Eroglu, and Cinar (2023)](https://link.springer.com/article/10.1007/s42044-023-00139-8) suggest within each class of tumor there are two types benign and malignant. Which describe if a tumor is essentially harmless or cancerous. Furthermore they describe gliomas as arising from neuroglial cells, meningiomas arising from brain membranes, and pituitary tumors are on pituitary glands.


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

## Problem_Statement:
Because of the difficulty of manual classification of a brain tumor and the time constraints put on radiologist, building a classification model would greatly reduce the workload of a radiologist. This model will help guide the MRI process by classifying the images so the radiologist will have a reference point when evaluating an MRI scan. This model should only be used as an assistant and not for diagnosis.

For the purposes of this model, we want to limit our false negatives. We want to avoid the model predicting no tumor when in reality there is a tumor. A false positive,predicting there is a tumor when there is no tumor, would simply require a follow up from a radiologist or specialized professional to confirm a diagnosis. In this case the model cares more about the precision and accuracy score of the model.

## EDA

For our [EDA process](./Notebooks/01_EDA.ipynb) we examined our two datasets which contained the classes 'glioma', 'meningioma', 'notumor', 'pituitary' which were separated into their own folders. The file count of the images were as follows:

|Class|Training count| Testing count|
|------|------|------|
|No Tumor|1,595|405|
|Meningioma Tumor|1,339|306|
|Glioma Tumor|1,321|300|
|Pituitary Tumor|1,457|300|


For our EDA process we took the average pixel value of each class.
Overall, the classes are balanced between the classes 'glioma', 'meningioma', 'notumor', and 'pituitary' in both our training and testing datasets. Additionally when looking at the average pixel value and the contrast between those average we saw some distinct features within those classes. This indicates that there are features that our model can learn to distinguish images into classes. One interesting observation we made was brain with no tumors generally had the most details. When reflecting on our research, [Morgan (2022)](https://www.cancer.gov/rare-brain-spine-tumor/blog/2022/neuroradiology) suggested the first MRI scan is used as a baseline to determine what areas of the brain to scan. This suggest a brain with no tumor will generally have the same angle across patients which was shown in our analysis.

## Modeling_and_Evaluations

For our [modeling stage](./Notebooks/03_Neural_network.ipynb) we built an initial [baseline model](./Notebooks/02_Baseline_Model.ipynb) that takes the average pixel values of a class. This method was used as a baseline to compare our model and achieved an accuracy of 46%. Following our baseline model we built 2 convolutional neural network, one as a simple baseline and another with regularization; however, the regularization had a poorer performance compared to out model without regularization. A 3rd convolutional neural network using augmentation was attempted, but there were issues with running out of memory so that idea was scrapped.

Our models were evaluated in the [Model Evaluation notebook](./Notebooks/04_Model_evaluation.ipynb) using a custom module to extract the accuracy, precision, recall, and F1 scores. the scores are as follows:

|Model|Accuracy|Precision|Recall|F1|
|-------|-------|--------|------|--------|
|Baseline|42%|46%|46%|46%|
|Neural Network: No regularization|87%|87%|87%|86%|
|Neural Network: with Regularization|84%|84%|84%|82%|

When looking at additional metrics we have more insight on the different metrics on individual classifications. In this case we only want to focus only on the no tumor class in which case we want to limit our false negative, so the metric we will focus on is Precision. In this case our Neural network with no regularization does the best across all metrics. The only mis-classification for the no tumor class is meningioma in which case we see 64 false negatives.

## Conclusion

Overall our best model has similar accuracy (87%) to manual classification from the radiologist found in the study by [Munir, S. Khan, Hanif, and M. Khan (2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7794124/) so our model could greatly help reduce the workload of radiologist. We primarily care about the false negative rate of our no tumor class and found the model to often predict meningioma incorrectly. [According to Zhang et al. (2022)](https://www.frontiersin.org/articles/10.3389/fonc.2022.886968/full) meningiomas are often benign and not as a severe as a risk as the other two classes of tumors. However, it still the risk of the missing a cancerous meningoma still remains, so that risk is still present.


## Recommendations

This model can be improved by identifying the type of machine used to perform the MRI. That way we can account for the variability between machines and technicians. We will need to talk to the data team to add additional labels.

To deal with data leakage we will need to talk to experts to identify and label only the initial image to avoid the varying types of imaging being misclassified. Ideally, we would only  utilize the first baseline image, but again we need to consult of domain experts to understand what image would cause data leakage.

This model can be combined with anomaly detection so that the tumor can be detected as well as be able to detect weather a tumor has grown in size. The data science team will need to consult with other teams to see if we have the available data and with administrators to see if this is a path we want to pursue.

In the future the streamlit app can be use to dynamically evaluate models by allowing users to upload a saved model. However at the moment that were issues with the user input field not passing the object as the a h5 model. This way radiologist and domain experts can correct the classifications, so the weights can be updated to be more accurate.

## Limitations

[Morgan (2022)](https://www.cancer.gov/rare-brain-spine-tumor/blog/2022/neuroradiology) brings up the argument of variation in MRI scans depending on the machine used. Because we don't know what machine was used to get these MRI scans, we cant account for variations in these images. The most significant difference is the resolution of images between machines, which will affect how well our model learns details. This fact is expended on by [Zacharki et al. (2009)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2863141/) which explains brain scans are often difficult to distinguish between the tissues of the brain. So in essence the quality of the machine has varying levels of image clarity which would influence our models performance. One area to EDA area to explore is to see the average pixel density of the images or identify what machine was used for the MRI scans.
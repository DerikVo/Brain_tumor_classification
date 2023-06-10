import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

'''
This portion uses code from a previous project from this [notebook](https://github.com/DerikVo/DSI_project_4_plant_disease/blob/main/notebooks/01_Potato_PlantVillageEDA.ipynb).
The code was originally developed by chat GPT 4 with the prompt:  "I have an image data set that I want to do EDA on. How can I average out the pixel values of all the images in a class. python keras."

This function takes two arguments the dataset: training or testing, and the sub_folder for the type of tumor e.g. ['glioma', 'meningioma', 'notumor', 'pituitary']
This function is used to find the average pixel values of each class
The purpose is to find if there is a difference in each class

'''
def avg_images(class_name, dataset='Training'):
    '''
    This function is used to find the average pixel value of each class
    
    Users will need to assign the images to a variable. 
    For example:
    meningioma_tumor = eda.avg_images('meningioma')
    =============================================================================    
    Keyword arguments:
    folders -- The sub folder containing the classifcation for tumor type ( 'glioma',
    'meningioma', 'notumor', 'pituitary' )
    dataset -- The main folder either Train and Test folder (default = Training)
    '''
    path = f'../Images/{dataset}/{class_name}/'
    image_files = os.listdir(path)
    num_images = len(image_files)
    average_image = np.zeros((256, 256, 1), dtype=np.float32)

    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        image = load_img(image_path, color_mode='grayscale', target_size=(256, 256))
        image_array = img_to_array(image)
        average_image += image_array / num_images

    return average_image

'''
The Code was originally developed by chat GPT 3 with the prompt: "I want to find the average pixel value of each class and then use the mean of an image to find which class it belongs to. The path to the class looks like '../Images/Training/glioma/' the classes are 'glioma', 'meningioma', 'notumor', 'pituitary'"

was later prompted to adjust the code to be able to pass a parameter to the classify_images function. Took a total of 8 prompts and  3 manual adjustments.

commented out the code to try to undestand what is happening so I can adjust the code as needed.
'''
def find_closest_class(mean_pixel_value, class_averages):
    '''
    This function finds which class an image is closest to
    =============================================================================    
    Keyword arguments:
    mean_pixel_value -- The mean pixel value of an image
    class_averages -- The main folder either Train and Test folder (default = Training)
    '''
    #initialize the cloest class variable
    closest_class = None
    #initialize the cloest class variable
    closest_distance = float('inf')
    for class_name, average in class_averages.items():
        #finds the distance between the mean pixel value and the class average
        #https://www.educative.io/answers/what-is-the-nplinalgnorm-method-in-numpy
        distance = np.linalg.norm(mean_pixel_value - average)
        # Finds the smaller distance
        if distance < closest_distance:
            #update the closest distance
            closest_distance = distance
            #updates the closest class
            closest_class = class_name

    return closest_class

'''
The Code was originally developed by chat GPT 3 with the prompt: "How do I dynamically classify images using the folder they are in as a class. Please use the OS module"

was later prompted to adjust the code to be able to pass a parameter to the find_closest_class function.

commented out the code to try to undestand what is happening so I can adjust the code as needed.
'''
def classify_images(test_folder_path, class_paths):
    '''
    This function finds predicts which class belongs to based on which avg pixel value its closest to
    =============================================================================    
    Keyword arguments:
    test_folder_path -- The file path to the test folder
    class_paths -- The file path to the class
    '''
    #create a list for the actual classes of an images
    actual_classes = []
    #create a list for the predictions
    predicted_classes = []
    #store the average pixel values for each class
    class_averages = {}
    for class_name, class_path in class_paths.items():
        #calculates the average pixcel value for each class (function should default to the training data set)
        average_image = avg_images(class_name)
        #finds the mean pixel value and stores it as a key value pair
        class_averages[class_name] = np.mean(average_image)

    for class_name, class_path in class_paths.items():
        #gets the path to the class folders under the testing data set
        class_folder_path = os.path.join(test_folder_path, class_name)
        #gets a list of the images within the folder
        image_files = os.listdir(class_folder_path)

        for image_file in image_files:
            #gets the path an individual image
            image_path = os.path.join(class_folder_path, image_file)
            #reads the image path using Open CV
            test_image = cv2.imread(image_path)
            #gets the mean pixel value of the image
            mean_pixel_value = np.mean(test_image, axis=(0, 1))
            #uses the find_cloest_class function to find what class its closest to
            closest_class = find_closest_class(mean_pixel_value, class_averages)
            #appends the actual class to the actual classes list
            actual_classes.append(class_name)
            #appends the predicted class to the predicted classes list
            predicted_classes.append(closest_class)

    return actual_classes, predicted_classes
'''
This portion reuses code from prior projects. The confusion matrix used the project: https://github.com/DerikVo/DSI_project_4_plant_disease/blob/main/notebooks/02_plant_village_potato_modeling.ipynb
which prompted ChatGPT 4 to help grab the labels information from the validation dataset and get it into a numpy array, so that we can use that to make a confusion matrix.

Converting a dictionary was taken from: https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe

The creatation of the data frame was taken from this project: https://github.com/DerikVo/NN_hackathon/blob/main/Code/Training/pre-trained-models.ipynb

portions of the code have been adapted to work with file pathways.

Prompted ChatGPT 3 with incorporating the code as a function that uses the classify_images function to get the confusion matrix and classification metrics. Then prompted and correct some syntax errors. 
'''

def calculate_metrics(actual_classes, predicted_classes, class_paths):
    '''
    This function calculates the precisision, recall, and F1 scores in the for of a data frame.
    This funcsion also creats a confusion matrix.
    =============================================================================    
    Keyword arguments:
    actual_classes -- The actual class an image belongs to
    predicted_classes -- The predicted class an image belongs to
    class_paths -- The file path to the class
    '''
    #creates the confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes,
    # gets the label of each class
    labels=list(class_paths.keys()))
    #Finds the weighted scores for each metric
    accuracy = accuracy_score(actual_classes, predicted_classes)
    precision = precision_score(actual_classes, predicted_classes, average='weighted')
    recall = recall_score(actual_classes, predicted_classes, average='weighted')
    f1 = f1_score(actual_classes, predicted_classes, average='weighted')
    #adds the scores into a data frame
    data = {'Accuracy': [accuracy],'Precision': [precision], 'Recall': [recall], 'F1 Score': [f1]}
    metrics_df = pd.DataFrame(data, index=['baseline'])

    return cm, metrics_df

'''
This portion reuses code from prior projects. The confusion matrix used the project: https://github.com/DerikVo/DSI_project_4_plant_disease/blob/main/notebooks/02_plant_village_potato_modeling.ipynb
which prompted ChatGPT 4 to help grab the labels information from the validation dataset and get it into a numpy array, so that we can use that to make a confusion matrix.

There were slight modifications to fit the purposes of this code such as assigning class paths and a title parameter.
'''
def plot_confusion_matrix(confusion_matrix, class_paths, title):
    '''
    This function displays the actual confusion matrix and plots it
    =============================================================================    
    Keyword arguments:
    confusion_matrix -- The confusion matrix generated from the calculate_metrics function
    class_paths -- The file path to the class
    title -- The title of the confusion matrix as well as the title of the saved image
    The tile will be a prefix to {title} confusion matrix
    '''
    #sets the figure size
    plt.figure(figsize=(10,10))
    #Plots the confusion matrix and assigns the class names on the axis ticks
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g',
    xticklabels=list(class_paths.keys()), yticklabels=list(class_paths.keys()))
    #labels the axis
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    #sets the title
    plt.title(f'{title} Confusion Matrix')
    #saves the figure
    plt.savefig(f'../Created_images/{title} confusion matrix.png')
    #displays the image
    plt.show()

'''

Creating a the data frame borrowed ideas from a one day hack-a-thon
https://github.com/DerikVo/NN_hackathon

Converting a dictionary was taken from: https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe

originally tried using functions that's only found in pandas 2.0

'''
def model_metrics(true_classes, predicted_classes, title):
    '''
    Calculate accuracy, precision, recall, and F1 score and returns a dataframe.
    Also passes a title argument that titles the index for the model being used
    '''
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    data = {'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1 Score': [f1]}
    df = pd.DataFrame(data, index=[f'{title}'])
    return df

from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import os
import random
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''
The concept of this code was learned from General Assembly's Data science Immersive's Excel lab excercise
The concepts has been adapted to identify file paths ways to work with the data
''' 
def folders(dataset = 'Training'):
    '''
    Finds the folders within either the 'Training' dataset or the 'Testing' data set.
    =============================================================================    
    Keyword arguments:
    dataset -- The main folder either Train and Test folder (default = Training)
    
    =================================================
    Example:
    train_folders = eda.folders(train_dataset)
    test_folders = eda.folders(test_dataset)
    '''
    #sets the path to the Testing and training folders
    path = f'../Images/{dataset}'
    #lists the classification folders within the dataset
    folders= os.listdir(path)
    return folders


'''

The concept of this code was learned from General Assembly's Data science Immersive's Excel lab excercise during the 2023-03 to 2023-06 cohort.
The concepts has been adapted to identify file paths ways to work with the data
'''
def image_len(folders, dataset ='Training'):
    '''
    
    List subfolders with the main folder containing the classification folders for each image set. As well as shows a random image from the classification
    =============================================================================    
    Keyword arguments:
    folders -- The sub folder containing the classifcation for tumor type ( 'glioma', 'meningioma', 'notumor', 'pituitary' )
    dataset -- The main folder either Training and Testing folder (default = Training)
    
    ===============================================
    Example:
    eda.image_len(train_folders)
    '''
    #sets the path to the Testing and training folders
    path = f'../Images/{dataset}'
    #loop through each classification folder
    for i in folders:
        #get number of images of folder
        num = len(os.listdir(f'{path}/{i}'))
        #prints the folder name and how many images are in each class
        print(f"{i} has {num} files")
        #get a random number to pick a random image from the folder
        rand = random.randint(0, num-1)
        #Gets the name of the actual image
        image_name = (os.listdir(f'{path}/{i}')[rand])
        #assigns the file path to the image
        image = load_img(f'{path}/{i}/{image_name}')
        #shows the image for the classification for reference
        plt.title(f'{image_name}')
        plt.imshow(image)
        plt.axis('off')
        plt.show()
'''
This portion uses code from a previous project from this [notebook](https://github.com/DerikVo/DSI_project_4_plant_disease/blob/main/notebooks/01_Potato_PlantVillageEDA.ipynb).
The code was originally developed by chat GPT 4 with the prompt:  "I have an image data set that I want to do EDA on. How can I average out the pixel values of all the images in a class. python keras."

This function takes two arguments the dataset: training or testing, and the sub_folder for the type of tumor e.g. ['glioma', 'meningioma', 'notumor', 'pituitary']
This function is used to find the average pixel values of each class
The purpose is to find if there is a difference in each class

'''
def avg_images(folders, dataset = 'Training'):
    '''

    This function is used to find the average pixel value of each class
    
    Users will need to assign the images to a variable. 
    For example:
    meningioma_tumor = eda.avg_images('meningioma')
    =============================================================================    
    Keyword arguments:
    folders -- The sub folder containing the classifcation for tumor type ( 'glioma', 'meningioma', 'notumor', 'pituitary' )
    dataset -- The main folder either Train and Test folder (default = Training)
    '''
    #sets the path to the Testing and training folders
    path = (f'../Images/{dataset}')
  
    class_name = folders
    batch_size = 32  # Modify this to suit your needs
    #instantiate ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255)  # normalize pixel values to [0,1]
    #get the images from the directory
    generator = datagen.flow_from_directory(path,
                                          classes=[class_name],
                                          class_mode=None,
                                          color_mode='grayscale',
                                          target_size=(256, 256),
                                          batch_size=batch_size)
    n_samples = generator.samples
    average_image = np.zeros((256, 256, 1))

    for i in range(n_samples // batch_size):  # Integer division to avoid partial batches
        images = next(generator)
        average_image += np.sum(images, axis=0)

    average_image /= n_samples
    return average_image

'''
This portion uses code from a previous project from this [notebook](https://github.com/DerikVo/DSI_project_4_plant_disease/blob/main/notebooks/01_Potato_PlantVillageEDA.ipynb). The concept was originally developed by [Yasser Siddiqui](syasser126@gmail.com) and has been adapted to use with this notebook.

This function is used to find the differences of the average pixel value between each class 'glioma', 'meningioma', and 'pituitary' compared to 'notumor'. 
These different characteristics can help us understand how the classes are unique when compared to not having a tumor.
If there are significant differences we can better interpret our model.
'''
def image_contrast(comparision, base_image):
    '''
    This function finds the differences between the pixel averages of two classes to identify how the model can differentiate classes
    Users will need to have ran the avg_images function for each class before running the image_contrats function.
    
    Users will need to assign the images to a variable. 
    For example:
    meningioma_contrast = eda.image_contrast(meningioma_tumor, notumor)
    =============================================================================    
    Keyword arguments:
    comparision -- The image that contains the type of tumor ( 'glioma', 'meningioma', 'pituitary' )
    base_image -- The image youre comparing against ('notumor')
    '''
    # we need to rescale the contrasts
    image = base_image - comparision
    # subtract minimum 
    image -= image.min()
    # divide by new max
    image /= image.max()

    return image
'''
refernced resources for colorblindness accomadation:
https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html#colorblindness

'''
def display_image(image, title):
    '''

    This function is to plot the three of the same images with different supports for varying color blindness 
    This function can be used for both the image_contrast and avg_images funcitons.
    
    Users will need to assign the images to a variable. 
    For example:
    meningioma_contrast = eda.image_contrast(meningioma_tumor, notumor)
    =============================================================================    
    Keyword arguments:
    Image -- The image you want to display
    Title -- The title of the image. This will be used for saving the figure as well
    '''
    
    #sets up the figure for the subplots
    fig, ax = plt.subplots(1,3, figsize = (10,10))
    #plots the title for the color map
    plt.suptitle(f'Pixel average: {title}', y = .75, fontsize = 22)
    #plotting the images
    ax[0].imshow(image)
    #uses the default Viridis color map; default colorblind friendly
    ax[0].set_title('Viridis', fontsize=18)
    #turns of axis bcause we only want the image
    ax[0].axis('off')
   
    #plotting the images
    ax[1].imshow(image, cmap='cividis')
    #plots the title for the color map
    ax[1].set_title('cividis', fontsize=18)
    #turns of axis bcause we only want the image
    ax[1].axis('off')
    
    #plotting the images
    ax[2].imshow(image, cmap='magma')
    #plots the title for the color map
    ax[2].set_title('magma', fontsize=18)
    #turns of axis bcause we only want the image
    ax[2].axis('off')
    
    plt.tight_layout()
    #saves the image
    plt.savefig(f'../Created_images/{title}.png')
    #displays the image in the notebook
    plt.show()

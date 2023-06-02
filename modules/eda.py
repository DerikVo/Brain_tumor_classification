from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import os
import random
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
from tensorflow.keras.utils import set_random_seed
set_random_seed(42)

def folders(dataset):
    path = f'../Images/{dataset}'
    images_dir = [x for x in os.listdir(path)]
    folders= os.listdir(path)
    return folders

def avg_images(dataset, sub_folder):
  '''
  This function takes two arguments the dataset: training or testing, and the sub_folder for the type of tumor e.g. ['glioma', 'meningioma', 'notumor', 'pituitary']
  This function is used to find the average pixel values of each class
  The purpose is to find if there is a difference in each class
  '''
  #assign the path in the function for readability and understanding
  #assign the sub folder (class name) that was passed to the function
  path = (f'../Images/{dataset}')
  class_name = sub_folder
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


def image_len(dataset, folders):
    '''
    This code takes in the list directory of the folder containing the classification folders. And the dataset.
    this code was heavily inspired by this project: https://github.com/DerikVo/DSI_project_4_plant_disease/blob/main/notebooks/01_Potato_PlantVillageEDA.ipynb
    Has since been adapted to work with a jupyter notebook
    TODO:convert all image eda into a class/method script
    '''
    #loop to each sub folder so we can get the class sizes
    path = f'../Images/{dataset}'
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
        #shows the image
        plt.title(f'{image_name}')
        plt.imshow(image)
        plt.axis('off')
        plt.show()
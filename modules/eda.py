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
    Finds the folders within either the Training dataset or the Testing data set.
    =============================================================================    
    Keyword arguments:
    dataset -- The main folder either Train and Test folder (default Training)
    '''
    path = f'../Images/{dataset}'
    images_dir = [x for x in os.listdir(path)]
    folders= os.listdir(path)
    return folders


'''

The concept of this code was learned from General Assembly's Data science Immersive's Excel lab excercise
The concepts has been adapted to identify file paths ways to work with the data
'''
def image_len(folders, dataset ='Training'):
    '''
    
    List subfolders with the main folder containing the classification folders for each image set.
    =============================================================================    
    Keyword arguments:
    folders -- The sub folder containing the classifcation for tumor type
    dataset -- The main folder either Train and Test folder (default Training)
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
'''
This function takes two arguments the dataset: training or testing, and the sub_folder for the type of tumor e.g. ['glioma', 'meningioma', 'notumor', 'pituitary']
This function is used to find the average pixel values of each class
The purpose is to find if there is a difference in each class
'''
def avg_images(folders, dataset = 'Training'):
  '''

  This function is used to find the average pixel value of each class
  =============================================================================    
  Keyword arguments:
  folders -- The sub folder containing the classifcation for tumor type
  dataset -- The main folder either Train and Test folder (default Training)
  '''
  #assign the path in the function for readability and understanding
  #assign the sub folder (class name) that was passed to the function
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
    
def image_contrast(comparision, base_image):
    # we need to rescale the contrasts
    image = base_image - comparision
    image -= image.min() # subtract minimum 
    image /= image.max() # divide by new max

    return image

def display_avg(avg_images, title):
#plotting the contrast between GLioma and no Tumor
    fig, ax = plt.subplots(1,3, figsize = (10,10))
    plt.suptitle(f'Pixel average: {title}', y = .75, fontsize = 22)
    ax[0].imshow(avg_images)
    ax[0].set_title('Viridis', fontsize=18)
    ax[0].axis('off')

    ax[1].imshow(avg_images, cmap='seismic')
    ax[1].set_title('Seismic', fontsize=18)
    ax[1].axis('off')

    ax[2].imshow(avg_images, cmap='BrBG')
    ax[2].set_title('BrBG', fontsize=18)
    ax[2].axis('off')
    plt.tight_layout()
    plt.savefig(f'../Created_images/Pixel_average_{title}.png')
    plt.show()

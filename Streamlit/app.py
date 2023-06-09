import streamlit as st
import numpy as np
import os
import pandas as pd

from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import time
import sys
sys.path.append('../modules/')
import model as m

_ = """
All comments will be assigned to the underscore variable so they dont get rendered in streamlit
as mention in this discussion form:
https://discuss.streamlit.io/t/any-way-to-prevent-commented-out-code-via-triple-quotes-to-be-displayed-in-streamlit/8821/6

This code takes heavy influece from a previous project.
https://github.com/DerikVo/NN_hackathon

There were many changes to the code to get it to work with this data set,
but the general structure remains the same
"""


# function to load and cache pretrained model
@st.cache_resource
def load_model_stream():
    path = "../Models/CNN_base.h5"
    model = load_model(path)
    return model

# function to preprocess an image and get a prediction from the model
def get_prediction(model, image):
	open_image = Image.open(image)
	resized_image = open_image.resize((256, 256))
	grayscale_image = resized_image.convert('L')
	img = np.expand_dims(grayscale_image, axis=0)
	predicted_prob = model.predict(img)[0]
	classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
	probabilities = dict(zip(classes, predicted_prob))
	sorted_probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
	return sorted_probabilities
def upload_mode():

  st.header("Classification Mode")
  st.subheader("Upload an Image to Make a Prediction")

  # upload an image
  uploaded_image = st.file_uploader("Upload your own image to test the model:", type=['jpg', 'jpeg', 'png'])

  # when an image is uploaded, display image and run inference
  if uploaded_image is not None:
    st.image(uploaded_image)
    st.text(get_prediction(classifier, uploaded_image))

st.set_page_config(layout="wide")

# load model
classifier = load_model_stream()

st.title("Brain Tumor Classifier")

st.write('Use the sidebar to select a page to view.')

page = st.sidebar.selectbox('Select Mode',['Upload Image','Model Evaluation']) 

_ ='''
This portion of the code was taken from the moduels function py file
this code also brows ideas from previous projects and intergrates it into a function.
Espically the model evaluation notebook.
'''
def model_Evaluation(path):
	'''
	Calculate accuracy, precision, recall, and F1 score.
	'''
	model = keras.models.load_model(path)
	testing_folder_path = '../Images/Testing'
	datagen = ImageDataGenerator()
	test_ds = datagen.flow_from_directory(
		testing_folder_path,
		target_size=(256, 256),
		color_mode='grayscale',
		class_mode='categorical',
		seed=42,
		shuffle=False
		)	
	true_classes = test_ds.classes
	y_pred = model.predict(test_ds)
	predicted_classes = np.argmax(y_pred, axis=1)
	accuracy = accuracy_score(true_classes, predicted_classes)
	precision = precision_score(true_classes, predicted_classes, average='weighted')
	recall = recall_score(true_classes, predicted_classes, average='weighted')
	f1 = f1_score(true_classes, predicted_classes, average='weighted')
	data = {'Accuracy': round(accuracy,4), 'Precision': round(precision,4), 'Recall': round(recall,4), 'F1 Score': round(f1,4)}
	return data
_ ='''
This code utilized the streamlit documentation to implement columns
and displaying images. In the future I want users to be able to upload their model and have it
automatically be evaluated by the app.

The links are as folows:
https://docs.streamlit.io/library/api-reference/layout/st.columns
https://docs.streamlit.io/library/api-reference/media/st.image
'''

if page == 'Model Evaluation':
	path = ('../Models/CNN_base.h5')
	data = model_Evaluation(path)
	
	reg_path = ('../Models/CNN_regularization.h5')
	reg_data = model_Evaluation(path)
	st.write("CNN base Metrics:\n")
	
	col1, col2 = st.columns(2)
	with col1:
		for metric_name, metric_value in data.items():
			st.write(f"{metric_name}: {metric_value}")
	with col2:
		st.image("../Created_images/Neural Network confusion matrix.png", caption = "No regularization")
	#add white space
	st.write("")
	st.write("")
	st.write("\n CNN regularization Metrics:")
	col3, col4 = st.columns(2)
	with col3:
		for metric_name, metric_value in reg_data.items():
			st.write(f"{metric_name}: {metric_value}")
	with col4:
		st.image("../Created_images/Neural Network with regularization confusion matrix.png", caption = "With regularization")

else:
  upload_mode()

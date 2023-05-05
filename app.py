import streamlit as st
import keras.backend as K
import numpy as np
import scipy.io
import cv2 as cv
from keras.models import load_model

st.title('Vehicle Price Prediction challenge')

model = load_model('/path/to/your/model.hdf5')
model.load_weights('/path/to/your/weights.hdf5')
cars_meta = scipy.io.loadmat('/path/to/cars_meta.mat')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)

def predict(image):
    img_width, img_height = 224, 224
    bgr_img = cv.imread(image)
    bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    preds = model.predict(rgb_img)
    prob = np.max(preds)
    class_id = np.argmax(preds)
    class_name = class_names[class_id][0][0]
    return class_name, prob



uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Make a prediction
    class_name, prob = predict(uploaded_file)
    st.success('Prediction: {} (Confidence Score: {:.2f}%)'.format(class_name, prob * 100))

# Clear the Keras session to free up memory
K.clear_session()

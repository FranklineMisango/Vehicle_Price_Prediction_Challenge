import streamlit as st
import keras.backend as K
import numpy as np
import tempfile
import os
import scipy.io
import cv2 as cv
from keras.models import load_model
from keras import initializers
import tensorflow as tf
from keras.utils import custom_object_scope
from keras.layers import Layer
from keras import backend as K
from keras.layers import InputSpec
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations


class Scale(Layer):
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        #self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        #self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_parameters = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out
        
    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def custom_layers():
    return {'Scale':Scale}
                    

# Load the model within the scope of custom_object_scope
with custom_object_scope(custom_layers()):
    model = load_model('/home/misango/code/Vehicle_Price_Prediction_Challenge/models/model.15-0.99.hdf5')

st.title('ResNet 152 & CART Vehicle Price Prediction challenge')
st.image('/home/misango/code/Vehicle_Price_Prediction_Challenge/images/st_image.jpg')
#model = load_model('/home/misango/code/Vehicle_Price_Prediction_Challenge/models/model.15-0.99.hdf5')
model.load_weights('/home/misango/code/Vehicle_Price_Prediction_Challenge/models/model.15-0.99.hdf5')
cars_meta = scipy.io.loadmat('/home/misango/code/Vehicle_Price_Prediction_Challenge/devkit/cars_meta.mat')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)


def predict(uploaded_file):
    test_path = ''
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        filename = temp_file.name
    bgr_img = cv.imread(filename)
    img_width, img_height = 224, 224
    bgr_img = cv.imread(filename)
    bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    preds = model.predict(rgb_img)
    prob = np.max(preds)
    class_id = np.argmax(preds)
    class_name = class_names[class_id][0][0]
    os.unlink(filename)
    return class_name, prob




uploaded_file = st.file_uploader('Upload a car image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Make a prediction
    class_name, prob = predict(uploaded_file)
    st.success('Prediction: {} (Confidence Score: {:.2f}%)'.format(class_name, prob * 100))

# Clear the Keras session to free up memory
K.clear_session()

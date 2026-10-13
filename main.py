import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Layer

class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        self.__dtype = dtype
        
    def call(slef, inputs):
        return tf.cast(inputs, self.__dtype)
    
model = tf.keras.models.load_model(
    "models/fer_25_vgg19.keras",
    compile=False,
    custom_objects={"Cast": Cast}
)
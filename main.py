import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Layer

# Placeholder Cast Layer
class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        self.__dtype = dtype
        
    def call(self, inputs):
        return tf.cast(inputs, self.__dtype)

# Load the model
model = tf.keras.models.load_model(
    "models/fer_25_vgg19.keras",
    compile=False,
    custom_objects={"Cast": Cast}
)

# Emotion labels
emotion_catagories = [
    'Anger', 'Disgust', 'Fear',
    'Happiness', 'Neutral', 'Sadness', 'Surprise'
]

# Prediction Function
def predict_emotion(img):
    if img is None:
        return {emotion: 0.0 for emotion in emotion_catagories}
    
    img = tf.convert_to_tensor(img)
    img = tf.cast(img, tf.float32)
    
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)
        
    img = tf.image.resize(img, (224, 224))
    
    img_array = tf.expand_dims(img, axis=0)
    
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)[0]
    
    return {emotion_catagories[i]: float(predictions[i]) for i in range(len(emotion_catagories))}

iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="Facial Emotion Recognition",
    description="Upload a face image and the model will predict the emotion"
)

if __name__ == "__main__":
    iface.launch()
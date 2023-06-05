import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Loading the Tensorflow Saved Model (PB)
model = tf.keras.models.load_model("saved_model")

# Saving the Model
tf.keras.models.save_model

pb_model_dir = "./auto_model/best_model"
h5_model = "./mymodel.h5"

# Loading the Tensorflow Saved Model (PB)
model = tf.keras.models.load_model(pb_model_dir)
print(model.summary())

# Saving the Model in H5 Format
tf.keras.models.save_model(model, h5_model)

# Loading the H5 Saved Model
loaded_model_from_h5 = tf.keras.models.load_model(h5_model)
print(loaded_model_from_h5.summary())
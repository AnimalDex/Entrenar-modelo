#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from tensorflow.keras.callbacks import TensorBoard
print("GPU Activada" if tf.config.list_physical_devices("GPU") else "Activa la GPU")


# In[2]:


dataset = "/home/usuario/data_modi"


# In[3]:


unique_breeds = []
for folders in os.listdir(dataset):
    breed = "".join(folders.split("-")[1:])
    unique_breeds.append(breed)
unique_breeds = np.array(sorted(unique_breeds))
print(unique_breeds)
print(unique_breeds.size)
len(unique_breeds), unique_breeds[:10]


# In[4]:


IMG_SIZE = 100
BATCH_SIZE = 32
EPOCAS = 100

#Image Data Pipeline Function
def image_data_pipeline(path, augment=False, img_size=IMG_SIZE, batch_size=BATCH_SIZE, test_data=False, seed=42):
    def retrieve_data_from_path(path, test_data=False):
        filenames = []
        if test_data:
            for files in os.listdir(path):
                filenames.append(f"{path}/{files}")
            return np.array(filenames)
        else:
            for folders in os.listdir(path):
                for files in os.listdir(f"{path}/{folders}"):
                    filenames.append(f"{path}/{folders}/{files}")
            filenames = np.array(filenames)
            np.random.shuffle(filenames)
            labels = np.array(["".join(name.split('/')[-2].split("-")[1:]) for name in filenames])
            unique_breeds = np.unique(labels)
            boolean_labels = np.array([label == unique_breeds for label in labels]).astype(int)
            return filenames, boolean_labels
    def process_image(filename, img_size=IMG_SIZE):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        #Convert colour channels values 0-255 to 0-1 values.
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[img_size, img_size])
        return image
    def configure_tensor(ds, shuffle=False):
        if shuffle:
            ds = ds.shuffle(buffer_size=1000) 
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        tf.keras.layers.RandomFlip(mode="horizontal")
    ])
    if test_data:
        print(f"Creating test data batches... BATCH SIZE={batch_size}")
        x = retrieve_data_from_path(path, test_data=True)
        x_test = tf.data.Dataset.from_tensor_slices(x).map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return configure_tensor(x_test)
    else:
        print(f"Creating train & validation data batches... BATCH SIZE={batch_size}")
        x, y = retrieve_data_from_path(path)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=seed)
        x_train = tf.data.Dataset.from_tensor_slices(x_train).map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        x_valid = tf.data.Dataset.from_tensor_slices(x_valid).map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if augment:
            x_train = x_train.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            x_valid = x_valid.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        y_train = tf.data.Dataset.from_tensor_slices(y_train)
        y_valid = tf.data.Dataset.from_tensor_slices(y_valid)
        train_data = tf.data.Dataset.zip((x_train, y_train)) 
        valid_data = tf.data.Dataset.zip((x_valid, y_valid)) 
        return configure_tensor(train_data, shuffle=True), configure_tensor(valid_data)


# In[5]:


train_data, valid_data = image_data_pipeline(dataset)
train_data.element_spec, valid_data.element_spec


# In[6]:


def show_batch(image_batch, label_batch):
  fig = plt.figure(figsize=(10,16))
  for n in range(BATCH_SIZE):
      ax = plt.subplot(8, 4, n + 1)
      plt.imshow(image_batch[n])
      plt.title(unique_breeds[label_batch[n]==1][0].title(), fontsize=12)
      plt.axis('off')
    
image_batch, label_batch = next(train_data.as_numpy_iterator())
show_batch(image_batch, label_batch)


# In[7]:


valid_image, valid_label = next(valid_data.as_numpy_iterator())
show_batch(valid_image, valid_label)


# In[8]:


#setup input shape into the model
INPUT_SHAPE = [None,IMG_SIZE, IMG_SIZE, 3] #batch, height, width, colour channel

#setup output shape of our model
OUTPUT_SHAPE = len(unique_breeds)

#setup model URL from Tensorflow HUB
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"


# Extract Pre-trained base Model and freeze the layers
base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5")
base_model.trainable = False


augment_train_data, augment_valid_data = image_data_pipeline(dataset, augment=True)
augment_train_image, augment_train_label = next(augment_train_data.as_numpy_iterator())
augment_val_image, augment_val_label = next(augment_valid_data.as_numpy_iterator())


# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(units=OUTPUT_SHAPE, activation="softmax")
])
model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
model.build(INPUT_SHAPE)
model.summary()
init_loss, init_acc = model.evaluate(valid_data)

tensorboardCNN = TensorBoard(log_dir='logs/modelo')
model.fit(
    x = train_data,
    epochs=EPOCAS, batch_size=32,
    validation_data=valid_data,
    validation_freq=1,
    callbacks=[tensorboardCNN],
)
base_loss, base_acc = model.evaluate(valid_data)
model.save('modelo.h5')

tensorboardCNN_AD = TensorBoard(log_dir='logs/modelo_ad')
model.fit(augment_train_data,
            epochs=EPOCAS,
            validation_data=augment_valid_data,
            validation_freq=1,
            callbacks=[tensorboardCNN_AD],
        )
aug_loss, aug_acc = model.evaluate(augment_valid_data)

model.save('modelo_ad.h5')


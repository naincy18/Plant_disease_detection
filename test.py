import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2 


model  = tf.keras.models.load_model('trained_model.keras')
model.summary()


# Visualizaing Single Image of Test set


image_path = '/Users/nainsikumari/Downloads/archive/test/test/0b8dabb7-5f1b-4fdc-b3fa-30b289707b90___JR_FrgE.S 3047_new30degFlipLR.JPG'
img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Convert BGR image to RGB
# img = img / 255.0

#Displaying Image
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()


# Testing Model


image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr]) #Convert single image to a batch
# input_arr = np.expand_dims(input_arr, axis=0)
# input_arr = input_arr / 255.0
print(input_arr.shape)



prediction = model.predict(input_arr)
prediction,prediction.shape

result_index = np.argmax(prediction)
result_index



class_name = ['Apple___Apple_scab',
             'Apple___Black_rot',
             'Apple___Cedar_apple_rust',
             'Apple___healthy',
             'Strawberry___Leaf_scorch',
             'Strawberry___healthy'
]

# train_datagen = ImageDataGenerator(
#     rescale=1/255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )






#Displaying Result of disease prediction
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()


model_prediction





from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

def createModel():

    train = ImageDataGenerator(rescale= 1/255)
    validation = ImageDataGenerator(rescale= 1/255)

    train_dataset = train.flow_from_directory('Images/train_dataset', target_size= (200,200), batch_size = 1000, class_mode = 'binary')
    validation_dataset = validation.flow_from_directory('Images/validation', target_size= (200,200), batch_size = 1000, class_mode = 'binary')

    model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape = (200,200,3)), 
    tf.keras.layers.MaxPool2D(2,2), 
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'), tf.keras.layers.MaxPool2D(2,2), 
    tf.keras.layers.Flatten(), tf.keras.layers.Dense(512,activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(loss='binary_crossentropy', optimizer = RMSprop(lr=0.001), metrics = ['accuracy'])

    model_fit=model.fit(train_dataset, steps_per_epoch = 50, epochs= 100, validation_data = validation_dataset)

    print(validation_dataset.class_indices)
    return(model_fit, model)

covid = createModel()

#Visualize the models accuracy
plt.plot(covid[0].history['accuracy'])
plt.plot(covid[0].history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', ' Val'], loc='upper left')
plt.savefig('plotimage/accuracy.png')

os.remove('plotimage/accuracy.png')


# #Visualize the models loss
# plt.plot(model_fit.history['loss'])
# plt.plot(model_fit.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', ' Val'], loc='upper right')
# plt.show()

 
# dir_path = 'Images/test_dataset'

# for i in os.listdir(dir_path):
#     img = image.load_img(dir_path+ '//' + i, target_size=(200,200))
#     print(i)
#     plt.imshow(img)
#     plt.show()

#     X = image.img_to_array(img)
#     X = np.expand_dims(X,axis =0)
#     images =np.vstack([X])
#     val = model.predict(images)

#     if val == 0:
#         print("Covid Positive")
#     else: 
#         print("Covid Negative")



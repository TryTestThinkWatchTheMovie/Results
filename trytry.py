import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.python.data import AUTOTUNE
from keras.models import load_model
from keras import layers
from keras.applications import EfficientNetB0
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.applications import imagenet_utils
from sklearn.metrics import classification_report
import imutils
from imutils import paths
import random
import cv2
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt


main_path = 'C:\\Users\\A\\Desktop\\Gun\\aDay\\trayDataset\\'
all_images = list(paths.list_images(main_path))

random_images = random.choices(all_images, k=3)

for i in random_images:
     random_image = cv2.imread(i)
     random_images = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
     random_image = imutils.resize(random_image, height=400)
     cv2.imshow("example", random_image)
     cv2.waitKey(0)
     cv2.destroyAllWindows()

random.shuffle(all_images)

i = int(len(all_images)*0.8) #80% to training
trainData = all_images[:i]#before the 80% of data
testData = all_images[i:]#after 80% of data

#validation data
i = int(len(trainData)*0.10) #10% for validation data
validData = trainData[:i]
trainData = trainData[i:]

train_path = main_path+'training'
test_path = main_path+'test'
valid_path = main_path+'valid'

datasets = [("training", trainData, train_path ), ("validation", validData, valid_path),  ("testing", testData, test_path)]

for (dtype, imagepaths, out_path) in datasets:
    if not os.path.exists(out_path):  # here we create the training, test and valid folders
        os.makedirs(out_path)

    for inputpath in imagepaths:
        filename = inputpath.split(os.path.sep)[-1]
        label = inputpath.split(os.path.sep)[-2]

        # Now that we have the labels, create the folder's path for each label for each train, test and validation folder
        labelPath = os.path.sep.join([out_path, label])

        if not os.path.exists(labelPath):
            os.makedirs(labelPath)  # create the folder we defined before (since we only had the path)

        # get the image and copy it to the respective label folder
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputpath, p)  # we take it from inputpath (the original images path) and move to the new folders


    def load_images(imagePath):
        # --images
        image = tf.io.read_file(imagePath)
        image = tf.io.decode_jpeg(image, channels=3)  # since there are 3 channels
        image = tf.image.resize(image, (150, 150)) / 255.0  # normalize the inputs
        # --labels
        label = tf.strings.split(imagePath, os.path.sep)[-2]
        label = tf.strings.to_number(label, tf.int32)
        print(label)
        return (image, label)


    def augment(image, label):
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image,0.2)  # defaul values. #You can use more augmentations techniques. Try them
        return (image, label)


trainPaths = list(paths.list_images(train_path))
valPaths = list(paths.list_images(valid_path))
testPaths = list(paths.list_images(test_path))

trainDS = tf.data.Dataset.from_tensor_slices(trainPaths)
    # Is important to add AUTOTUNE so tf takes care of all
trainDS = (trainDS.shuffle(len(trainPaths)).map(load_images, num_parallel_calls=AUTOTUNE).map(augment,num_parallel_calls=AUTOTUNE).cache().batch(16).prefetch(AUTOTUNE))
    # --validation
valDS = tf.data.Dataset.from_tensor_slices(valPaths)
valDS = (valDS.map(load_images, num_parallel_calls=AUTOTUNE).cache().batch(16).prefetch(AUTOTUNE))
# Remember not to shuffle and augment the validation and test sets, but as always we have to preprocess it

# --Test
testDS = tf.data.Dataset.from_tensor_slices(testPaths)
testDS = (testDS.map(load_images, num_parallel_calls=AUTOTUNE).cache().batch(16).prefetch(AUTOTUNE))


NUM_CLASSES = 2
IMG_SIZE = 150
size = (IMG_SIZE, IMG_SIZE,3) #add the 3rd channel

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

#let's use the model as a feature extractor

# include_top is set to false allowing a new output layer to be added and trained.
# we do this so we re-purpose this pretrained model to fit our task

model = EfficientNetB0(include_top=False, input_shape=size)

#add a new flatten layer that fits our task

flat1 = layers.Flatten()(model.layers[-1].output)
class1 = layers.Dense(1024, activation='relu')(flat1)
output = layers.Dense(1, activation='sigmoid')(class1) #use 1 output since is either evening or morning and use sigmoid since the probability is either 0 or 1

model = tf.keras.Model(inputs=model.inputs, outputs=output)

model.summary()


opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


early_s = EarlyStopping(monitor="val_loss", patience = 10, restore_best_weights=True)
save_b = ModelCheckpoint(filepath ="C:\\Users\\A\\Desktop\\Gun\\", monitor="val_loss", verbose = 1 )
callbacks = [early_s, save_b]


hist = model.fit(x = trainDS, validation_data=valDS, epochs= 50, callbacks=callbacks, verbose=1)


plt.figure()
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("loss/accuracy")
plt.legend(loc="lower left")
plt.show()


test_paths = list(paths.list_images(test_path))
testlabels = [int(p.split(os.path.sep)[-2]) for p in test_paths]
#turn them into categorical values so we can count them (one hot encoding)
testlabels = to_categorical(testlabels)


predictions = model.predict(testDS)

#1 is evening and 0 is morning

print(classification_report(testlabels.argmax(axis=1), predictions.argmax(axis=1), target_names=["0", "1"]))


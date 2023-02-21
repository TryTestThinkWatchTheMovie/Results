import numpy as np
from keras import Sequential
from keras.applications import EfficientNetB0
from keras.applications.densenet import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import tensorflow as tf

data_dir = '.../data/'

IMAGE_SIZE = [150, 150]
EPOCHS = 10
SEED = 42
CLASSES = 2
BATCH_SIZE = 16


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    color_mode='rgb',
    shuffle=True,
    class_mode='binary',
    subset="training"
)

# Test Generator - Augmentation
test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    color_mode='rgb',
    shuffle=True,
    class_mode='binary',
    subset="validation"
)


model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(150,150,3),classes=CLASSES)

flat1 = layers.Flatten()(model.layers[-1].output)
class1 = layers.Dense(1024, activation='relu')(flat1)
output = layers.Dense(1, activation='sigmoid')(class1) #use 1 output since is either evening or morning and use sigmoid since the probability is either 0 or 1

model = tf.keras.Model(inputs=model.inputs, outputs=output)

model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'binary_crossentropy', metrics = ["accuracy"])





hist = model.fit(
train_generator,
epochs = EPOCHS,
steps_per_epoch = train_generator.samples // BATCH_SIZE,
validation_data = test_generator,
validation_steps = test_generator.samples // BATCH_SIZE,
verbose = 1)


target_names = ['0','1']

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

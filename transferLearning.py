import os
from imutils import paths
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from matplotlib import pyplot
from sklearn.metrics import classification_report
batch_size= 32
image_shape = (224, 224)
dst_dir = '...\\morningTrayEveningTray\\'
train_datagen = ImageDataGenerator(rescale=1./255,
                                   brightness_range = (0.5,1.5),
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.15,
                                   zoom_range=0.1,
                                   channel_shift_range = 10,
                                   horizontal_flip=True,
                                   validation_split= 0.2
                                   )
test_datagen = ImageDataGenerator(rescale=1./255,
                                  validation_split= 0.2)

train_generator = train_datagen.flow_from_directory(dst_dir,
                                                    target_size=image_shape,
                                                    batch_size=batch_size,
                                                    seed = 835,
                                                    subset= "training",
                                                    class_mode = "binary"
                                                    )
val_generator = test_datagen.flow_from_directory(dst_dir,
                                                 target_size=image_shape,
                                                 batch_size=batch_size,
                                                 seed = 835,
                                                 subset="validation",
                                                 class_mode = "binary"
                                                 )
test_generator = test_datagen.flow_from_directory(
    dst_dir,
    target_size=image_shape,
    batch_size=batch_size,
    seed=835,
    color_mode='rgb',
    shuffle=True,
    class_mode='binary',
    subset="validation"
)


early_stopping = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                                   min_delta=0,
                                                   patience=2,
                                                   verbose=0,
                                                   mode="auto",
                                                   baseline=None,
                                                   restore_best_weights=False,
                                                   )
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                factor=0.1,
                                                patience=2,
                                                verbose=0,
                                                mode="auto",
                                                min_delta=0.0001,
                                                cooldown=0,
                                                min_lr=0,
                                                )
vgg16_model = tf.keras.applications.VGG16(pooling='avg',
                                          weights='imagenet',
                                          include_top=False,
                                          input_shape=image_shape +(3,)
                                          )
for layers in vgg16_model.layers:
            layers.trainable=False
last_output = vgg16_model.layers[-1].output
vgg_x = Flatten()(last_output)
vgg_x = Dense(128, activation = 'relu')(vgg_x)
vgg_x = Dense(7, activation = 'softmax')(vgg_x)
vgg16_model = tf.keras.Model(vgg16_model.input, vgg_x)
vgg16_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                          optimizer= 'adam',
                          metrics=['acc'])
#vgg16_final_model.summary()
vgg16_model.summary()


number_of_epochs = 30
vgg16_history = vgg16_model.fit(train_generator,
                                      epochs = number_of_epochs,
                                      validation_data = val_generator,
                                      callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)],
                                      verbose=1)




predictions = vgg16_model.predict(test_generator)

print(classification_report(test_generator.classes, predictions.argmax(axis=1), target_names=["0", "1"]))


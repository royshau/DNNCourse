import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from models import RESNET50,classification_model
from data_loader import create_dataset,generator_true_false

train_file_path = "data/train_relationships.csv"
train_folders_path = "data/train/"
val_famillies = "F09"
img_size = (197,197)

train, train_person_to_images_map,val,val_person_to_images_map = create_dataset(train_file_path,train_folders_path,val_famillies)

model = classification_model()
model.summary()
model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))
file_path = "resnet_22_06.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

model.fit_generator(generator_true_false(train, train_person_to_images_map,img_size, batch_size=16), use_multiprocessing=True,
                    validation_data=generator_true_false(val, val_person_to_images_map,img_size, batch_size=16), epochs=100, verbose=2,
                    workers=16, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)

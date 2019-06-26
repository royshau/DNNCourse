import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from models import RESNET50,classification_model,triplet_loss,triplet_loss_model,triplet_metric
from data_loader import create_dataset,generator_true_false,generator_triplets

train_file_path = "data/train_relationships.csv"
train_folders_path = "data/train/"
val_famillies = "F09"
img_size = (197,197)

train, train_person_to_images_map,val,val_person_to_images_map = create_dataset(train_file_path,train_folders_path,val_famillies)

model = triplet_loss_model()
model.summary()
model.compile(loss=triplet_loss, metrics=[triplet_metric], optimizer=Adam(0.00001))
file_path = "triple_24_06.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

# model.fit_generator(generator_true_false(train, train_person_to_images_map,img_size, batch_size=16), use_multiprocessing=True,
#                     validation_data=generator_true_false(val, val_person_to_images_map,img_size, batch_size=16), epochs=100, verbose=2,
#                     workers=16, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)

model.fit_generator(generator_triplets(train, train_person_to_images_map,img_size, batch_size=16), use_multiprocessing=True,
                    validation_data=generator_triplets(val, val_person_to_images_map,img_size, batch_size=16), epochs=100, verbose=2,
                    workers=16, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)
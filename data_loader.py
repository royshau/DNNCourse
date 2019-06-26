import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from collections import defaultdict
from glob import glob
import numpy as np
from random import choice, sample
import cv2

from keras_vggface.utils import preprocess_input
import os

def read_img(path,input_size):
    img = keras.preprocessing.image.load_img(path,target_size = input_size)
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

def generator_true_false(list_tuples, person_to_images_map,input_size, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x,input_size) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x,input_size) for x in X2])

        yield [X1, X2], labels

def generator_triplets(list_tuples, person_to_images_map,input_size, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size)

        for i in range(batch_size):
            while 1:
                p1 = batch_tuples[i][0]
                p3 = choice(ppl)

                if p1 != p3 and (p1, p3) not in list_tuples and (p3, p1) not in list_tuples:
                    batch_tuples[i] += (p3,)
                    break

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        anchor = np.array([read_img(x,input_size) for x in X])

        X = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        good = np.array([read_img(x,input_size) for x in X])

        X = [choice(person_to_images_map[x[2]]) for x in batch_tuples]
        bad = np.array([read_img(x,input_size) for x in X])
        labels = [1] * len(batch_tuples) #Dummy labels
        yield [anchor, good,bad] , labels


def create_dataset(relationships_file,images_path,validation_prefix):
    all_images = glob(os.path.join(images_path,'*/*/*.jpg'))

    train_images = [x for x in all_images if validation_prefix not in x]
    val_images = [x for x in all_images if validation_prefix in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    relationships = pd.read_csv(relationships_file)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if validation_prefix not in x[0]]
    val = [x for x in relationships if validation_prefix in x[0]]

    return train, train_person_to_images_map,val,val_person_to_images_map

# train_file_path = "data/train_relationships.csv"
# train_folders_path = "data/train/"
# val_famillies = "F09"
# train, train_person_to_images_map,val,val_person_to_images_map = create_dataset(train_file_path,train_folders_path,val_famillies)
#
# output = next(generator_triplets(train,train_person_to_images_map))
print("End test")
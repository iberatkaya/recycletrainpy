import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

imgSize = 120
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

cardboardData = tf.data.Dataset.list_files('testdata/cardboard/*')
glassData = tf.data.Dataset.list_files('testdata/glass/*')
metalData = tf.data.Dataset.list_files('testdata/metal/*')
paperData = tf.data.Dataset.list_files('testdata/paper/*')
plasticData = tf.data.Dataset.list_files('testdata/plastic/*')
trashData = tf.data.Dataset.list_files('testdata/trash/*')


def decode_img(img_data):
    img_data = tf.image.decode_jpeg(img_data, channels=1)
    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
    print(img_data)
    return tf.image.resize(img_data, [imgSize, imgSize])


def process_path(file_path):
    img_label = get_label(file_path.numpy())
    img_data = tf.io.read_file(file_path.numpy())
    img_data = decode_img(img_data)
    return img_data, img_label


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[2]


data = []
labels = []
numOfImages = 30
model = keras.models.load_model('model.h5')
for i in cardboardData.take(numOfImages):
    img, label = process_path('./data/cardboard/' + get_label(i.numpy()))
    data.append(img)
    labels.append(label)
for i in glassData.take(numOfImages):
    img, label = process_path('./data/glass/' + get_label(i.numpy()))
    data.append(img)
    labels.append(label)
for i in metalData.take(numOfImages):
    img, label = process_path('./data/metal/' + get_label(i.numpy()))
    data.append(img)
    labels.append(label)
for i in paperData.take(numOfImages):
    img, label = process_path('./data/paper/' + get_label(i.numpy()))
    data.append(img)
    labels.append(label)
for i in plasticData.take(numOfImages):
    img, label = process_path('./data/plastic/' + get_label(i.numpy()))
    data.append(img)
    labels.append(label)

y = []
for i in labels:
    if str(i.numpy().decode("utf-8")) == CLASS_NAMES[0]:
        y.append(0)
    elif str(i.numpy().decode("utf-8")) == CLASS_NAMES[1]:
        y.append(1)
    elif str(i.numpy().decode("utf-8")) == CLASS_NAMES[2]:
        y.append(2)
    elif str(i.numpy().decode("utf-8")) == CLASS_NAMES[3]:
        y.append(3)
    elif str(i.numpy().decode("utf-8")) == CLASS_NAMES[4]:
        y.append(4)
    elif str(i.numpy().decode("utf-8")) == CLASS_NAMES[5]:
        y.append(5)

print(y)

true = 0
false = 0
for i in range(len(data)):
    dataarr = tf.stack([data[i]])
    pred = model.predict(dataarr)
    if np.argmax(pred[0]) == y[i]:
        true += 1
    else:
        false += 1

print("True: " + str(true))
print("False: " + str(false))

import tensorflow as tf
import os
from tensorflow import keras
from random import shuffle
import tensorflowjs as tfjs


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

imgSize = 120
valSize = 300
val_batchSize = 128
batch_size = 256
numOfImages = 1500

cardboardData = tf.data.Dataset.list_files('data/cardboard/*')
glassData = tf.data.Dataset.list_files('data/glass/*')
metalData = tf.data.Dataset.list_files('data/metal/*')
paperData = tf.data.Dataset.list_files('data/paper/*')
plasticData = tf.data.Dataset.list_files('data/plastic/*')
trashData = tf.data.Dataset.list_files('data/trash/*')

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def decode_img(img_data):
    img_data = tf.image.decode_jpeg(img_data, channels=1)
    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
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

print('Creating tensors')

for i in cardboardData.take(numOfImages):
    img, label = process_path('./data/cardboard/' + get_label(i.numpy()))
    data.append({"img": img, "label": label})
for i in glassData.take(numOfImages):
    img, label = process_path('./data/glass/' + get_label(i.numpy()))
    data.append({"img": img, "label": label})
for i in metalData.take(numOfImages):
    img, label = process_path('./data/metal/' + get_label(i.numpy()))
    data.append({"img": img, "label": label})
for i in paperData.take(numOfImages):
    img, label = process_path('./data/paper/' + get_label(i.numpy()))
    data.append({"img": img, "label": label})
for i in plasticData.take(numOfImages):
    img, label = process_path('./data/plastic/' + get_label(i.numpy()))
    data.append({"img": img, "label": label})
for i in trashData.take(numOfImages):
    img, label = process_path('./data/trash/' + get_label(i.numpy()))
    data.append({"img": img, "label": label})

print('Created tensors')


shuffle(data)
x = []
y = []

for i in data:
    x.append(i["img"])
    if str(i["label"].numpy().decode("utf-8")) == CLASS_NAMES[0]:
        y.append(0)
    elif str(i["label"].numpy().decode("utf-8")) == CLASS_NAMES[1]:
        y.append(1)
    elif str(i["label"].numpy().decode("utf-8")) == CLASS_NAMES[2]:
        y.append(2)
    elif str(i["label"].numpy().decode("utf-8")) == CLASS_NAMES[3]:
        y.append(3)
    elif str(i["label"].numpy().decode("utf-8")) == CLASS_NAMES[4]:
        y.append(4)
    elif str(i["label"].numpy().decode("utf-8")) == CLASS_NAMES[5]:
        y.append(5)
    else:
        print('ERROR')

x_val = []
y_val = []

ctrs = [0, 0, 0, 0, 0, 0]
i = 0

while len(x_val) != valSize * 6:
    if ctrs[y[i]] != valSize:
        ctrs[y[i]] += 1
        x_val.append(x[i])
        y_val.append(y[i])
        del x[i]
        del y[i]
    i += 1

print(ctrs)

print(len(y))
print(len(y_val))

y_train = tf.one_hot(y, 6)
y_valTrain = tf.one_hot(y_val, 6)


train_dataset = tf.data.Dataset.from_tensor_slices((x, y_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_valTrain)).batch(val_batchSize)

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=64, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3),
                              input_shape=[imgSize, imgSize, 1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=64, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Conv2D(filters=128, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=128, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Conv2D(filters=256, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=256, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Conv2D(filters=512, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=512, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=512, strides=(2, 2), padding="same", activation='relu', kernel_size=(3, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(activation='relu', units=2048))
model.add(keras.layers.Dense(activation='softmax', units=6))

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=["categorical_accuracy", "categorical_crossentropy"]
)

model.summary()

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=300,
    verbose=1,
#    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_categorical_crossentropy', patience=12)]
)

model.save('model.h5')

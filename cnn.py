#%%
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math, datetime, platform, os

import os
for dirname, _, filenames in os.walk('inputData'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train = pd.read_csv('inputData/train.csv')
test = pd.read_csv('inputData/test.csv')

X = train.iloc[:, 1:785]
Y = train.iloc[:, 0]
X_test = test.iloc[:, 0:784]

""" X_tsn = X/255
from sklearn.manifold._t_sne import TSNE
tsne = TSNE()
tsne_res = tsne.fit_transform(X_tsn)
plt.figure(figsize=(14,12))
plt.scatter(tsne_res[:,0], tsne_res[:,1], c=Y, s=2)
plt.xticks([])
plt.yticks([])
plt.colorbar() """
# %%
from sklearn.model_selection import train_test_split
X_training, X_validation, Y_training, Y_validation = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state=1212
)

# %%

x_training_array = X_training.to_numpy().reshape(X_training.shape[0],28,28)
y_training_array = Y_training.values

x_validation_array = X_validation.to_numpy().reshape(X_validation.shape[0],28,28)
y_validation_array = Y_validation.values

x_testing_array = test.to_numpy().reshape(test.shape[0],28,28)

pd.DataFrame(x_training_array[0])
# %%

plt.imshow(x_training_array[0], cmap=plt.cm.binary)
plt.show

numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_training_array[i], cmap=plt.cm.binary)
    plt.xlabel(y_training_array[i])
plt.show()

# %%

x_training_channels = x_training_array.reshape(
    x_training_array.shape[0],
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS
)

x_validation_channels = x_validation_array.reshape(
    x_validation_array.shape[0],
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS
)

x_testing_channels = x_testing_array.reshape(
    x_testing_array.shape[0],
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS
)

x_training_normalized = x_training_channels / 255
x_validation_normalized = x_validation_channels / 255
x_testing_normalized = x_testing_channels / 255


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Convolution2D(
    input_shape=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS),
    kernel_size = 5,
    filters = 8,
    strides = 1,
    activation = tf.keras.activations.relu,
    kernel_initializer = tf.keras.initializers.VarianceScaling()
))

model.add(tf.keras.layers.MaxPooling2D(
    pool_size= (2,2),
    strides = (2,2)
))

model.add(tf.keras.layers.Convolution2D(
    kernel_size = 5,
    filters = 16,
    strides = 1,
    activation = tf.keras.activations.relu,
    kernel_initializer = tf.keras.initializers.VarianceScaling()
))

model.add(tf.keras.layers.MaxPooling2D(
    pool_size= (2,2),
    strides = (2,2)
))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(
    units=128,
    activation=tf.keras.activations.relu
))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(
    units = 10,
    activation = tf.keras.activations.softmax,
    kernel_initializer=tf.keras.initializers.VarianceScaling()
    
))

model.summary()
tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
)
# %%
adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

model.compile(
    optimizer=adam_optimizer,
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)
# %%
log_dir = ".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq = 1)

training_history = model.fit(
    x_training_normalized,
    y_training_array,
    epochs = 10,
    validation_data = (x_validation_normalized, y_validation_array),
    callbacks=[tensorboard_callback]
)

print("successful training complete")
# %%


plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['loss'], label='training set')
plt.plot(training_history.history['val_loss'], label='validation set')
plt.legend()

# %%
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(training_history.history['accuracy'], label='training set')
plt.plot(training_history.history['val_accuracy'], label='validation set')
plt.legend()

#%%
train_loss, train_accuracy = model.evaluate(x_training_normalized, y_training_array)
print('Train loss: ', train_loss)
print('Train accuracy: ', train_accuracy)
# %%
validation_loss, validation_accuracy = model.evaluate(x_validation_normalized, y_validation_array)
print('Validation loss: ', validation_loss)
print('Validation accuracy: ', validation_accuracy)
# %%
model_name = 'digit_recognizer.h5'
model.save(model_name, save_format='h5')
loaded_model = tf.keras.models.load_model(model_name)
# %%
predictions_one_hot = loaded_model.predict([x_validation_normalized])

predictions = np.argmax(predictions_one_hot, axis = 1)
pd.DataFrame(predictions)

plt.imshow(x_validation_normalized[0].reshape(
    (IMG_WIDTH, IMG_HEIGHT)), cmap=plt.cm.binary)

plt.show
# %%

numbers_to_display = 196
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(15,15))

for index in range(numbers_to_display):
    predicted_label = predictions[index]
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    color_map = 'Greens' if predicted_label == y_validation_array[index] else 'Reds'
    plt.subplot(num_cells,num_cells, index + 1)
    plt.imshow(x_validation_normalized[index].reshape((IMG_WIDTH,IMG_HEIGHT)), cmap=color_map)
    plt.xlabel(predicted_label)
plt.subplots_adjust(hspace=1, wspace=0.5)
plt.show()
# %%

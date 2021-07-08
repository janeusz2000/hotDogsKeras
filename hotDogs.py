# %%
import itertools
import matplotlib.image as mpimg
import PIL
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import numpy as np

print(device_lib.list_local_devices())

# %% PATH Acquisition
datasetPath = os.path.join(os.getcwd(), 'data')

trainPath = os.path.join(datasetPath, 'train')
train_hot_dog_path = os.path.join(trainPath, 'hot_dog')
train_numberOfHotDogs = len([file for file in os.listdir(train_hot_dog_path)])
train_not_hot_dog_path = os.path.join(trainPath, 'not_hot_dog')
train_numberOfNotHotDogs = len(
    [file for file in os.listdir(train_not_hot_dog_path)])


validationPath = os.path.join(datasetPath, 'validation')
validation_hot_dog_path = os.path.join(validationPath, 'hot_dog')
validation_numberOfHotDogs = len(
    [file for file in os.listdir(validation_hot_dog_path)])
validation_not_hot_dog_path = os.path.join(validationPath, 'not_hot_dog')
validation_numberOfNotHotDogs = len(
    [file for file in os.listdir(validation_not_hot_dog_path)])

testPath = os.path.join(datasetPath, 'test')

test_hot_dog_path = os.path.join(testPath, 'hot_dog')
test_not_hot_dog_path = os.path.join(testPath, 'not_hot_dog')
test_numberOfNotHotDogs = len(
    [file for file in os.listdir(test_not_hot_dog_path)])
test_numberOfHotDogs = len(
    [file for file in os.listdir(test_hot_dog_path)])

print(
    f"""Train hot dog path: {train_hot_dog_path}
number of files inside: {train_numberOfHotDogs}
train not hot dog path: {train_not_hot_dog_path}
number of files inside: {train_numberOfNotHotDogs}

Validation hot dog path: {validation_hot_dog_path}
number of files inside: {validation_numberOfHotDogs}
Train not hot dog path: {train_not_hot_dog_path}
number of files inside: {validation_numberOfNotHotDogs}

Validation hot dog path: {test_hot_dog_path}
number of files inside: {test_numberOfHotDogs}
Train not hot dog path: {test_not_hot_dog_path}
number of files inside: {test_numberOfNotHotDogs}""")


# %% training Properties
batchSize = 16
imageSize = (20, 20)

# %% data generators
train_image_augmentation = ImageDataGenerator(rescale=1./255)
train_generator = train_image_augmentation.flow_from_directory(
    trainPath,
    batch_size=batchSize,
    shuffle=True,
    target_size=imageSize,
    class_mode='binary')

train_generator_test = train_image_augmentation.flow_from_directory(
    trainPath,
    batch_size=batchSize,
    shuffle=False,
    target_size=imageSize,
    class_mode='binary')


validation_image_augmentation = ImageDataGenerator(rescale=1./255)
validation_generator = validation_image_augmentation.flow_from_directory(
    validationPath,
    shuffle=False,
    target_size=imageSize,
    class_mode='binary')

test_image_augmentation = ImageDataGenerator(rescale=1./255)
test_generator = test_image_augmentation.flow_from_directory(
    testPath,
    shuffle=False,
    target_size=imageSize,
    class_mode='binary'
)

# %% Model building
inputShape = (imageSize[0], imageSize[1], 3)
model = Sequential()
model.add(layers.Conv2D(3, (3, 3), activation='relu', input_shape=inputShape))
model.add(layers.Flatten(input_shape=inputShape))
model.add(layers.Dense(5, activation='tanh'))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(1, activation='sigmoid'))


model.summary()

# %%
model.compile(
    loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0005), metrics=["accuracy"]
)

# %% train
history = model.fit(
    x=train_generator,
    epochs=10,
    validation_data=validation_generator,
)
# %% evaluation
model.evaluate(test_generator)

# %%


def debugPredictions(train_generator_test):
    prediction = model.predict(train_generator_test)
    labels = np.reshape(train_generator_test.labels, [-1, 1])
    losses = keras.losses.binary_crossentropy(labels, prediction).numpy()
    data = sorted(zip(train_generator_test.filenames, losses,
                      prediction, labels), key=lambda x: -x[1])[:10]
    for row in data:
        print(row)
        img = mpimg.imread(train_generator_test.directory + f"/{row[0]}")
        plt.imshow(img)
        plt.show()


model.evaluate(validation_generator)
# %%
debugPredictions(train_generator)

# %%
path = train_generator_test.directory + "\\not_hot_dog\\118378.jpg"


def debugImage(xElements: int, yElements: int, path: str):
    from keras.preprocessing.image import load_img, img_to_array
    from keras.applications.vgg16 import preprocess_input
    from PIL.Image import ANTIALIAS
    img = load_img(path)
    img = img.resize(imageSize, ANTIALIAS)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    model.predict(img)
    xOffset = img.shape[1] // xElements
    yOffset = img.shape[2] // yElements

    xIndex = 0
    yIndex = 0
    while ((xIndex * xOffset) < img.shape[1]
           and (yIndex * yOffset) < img.shape[2]):
        image = np.copy(img)
        image[0, xIndex * xOffset:(xIndex + 1) * xOffset,
              yIndex * yOffset: (yIndex + 1) * yOffset, :] = 0
        xIndex += 1
        yIndex += 1
        print(f"index x {xIndex} index y {yIndex} offset: {xOffset}")
        plt.imshow(image[0, :, :, :] / 255.0)
        plt.show()

        print(model.predict(img))
        print("Should be not hotdog") if "not" in path else print(
            "Should be hotdog")


# %%
debugImage(10, 10, path)

# %%
print(history.history.keys())
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epoch")
plt.grid(True, which="both")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.xlabel("Epoch")
plt.grid(True, which="both")
plt.show()

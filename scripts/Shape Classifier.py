from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

# Constants
NUMBER_OF_CLASSES = 3
IMAGE_WIDTH, IMAGE_HEIGHT = 200, 200

# Build Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), padding="same", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(NUMBER_OF_CLASSES))
model.add(Activation("softmax"))

# Optimizer
optimizer = Adam()

# Compile Model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

# Fitting the Dataset
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_data_generator.flow_from_directory(
    directory="Dataset/training_set",
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=16,
    class_mode="categorical")

test_data_generator = ImageDataGenerator(rescale=1./255)

test_generator = test_data_generator.flow_from_directory(
    directory="Dataset/test_set",
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=16,
    class_mode="categorical")

model.fit_generator(train_generator, steps_per_epoch=100, epochs=5, validation_data=test_generator, validation_steps=100)

# Saving Model
model.save("shape_classifier_le_net_5.h5")

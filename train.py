import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout

print("Training started...")

data = []
labels = []

cur_path = os.getcwd()

# Dataset path
train_path = os.path.join(cur_path, 'dataset', 'Train')

# Get folders
folders = sorted(os.listdir(train_path))

print("Folders Found:", folders)

# Create label mapping
label_map = {}

for index, folder in enumerate(folders):
    label_map[folder] = index

print("Label Mapping:", label_map)

# Number of classes
classes = len(folders)

# Load dataset
for folder in folders:

    path = os.path.join(train_path, folder)

    if not os.path.isdir(path):
        continue

    images = os.listdir(path)

    print(f"Loading Folder {folder}")

    for img in images:

        try:
            image_path = os.path.join(path, img)

            image = cv2.imread(image_path)

            image = cv2.resize(image, (30, 30))

            image = np.array(image)

            data.append(image)

            # Use remapped labels
            labels.append(label_map[folder])

        except:
            print(f"Error loading image in folder {folder}")

# Convert to numpy
data = np.array(data)
labels = np.array(labels)

print("Dataset Loaded Successfully")
print("Data Shape:", data.shape)
print("Labels Shape:", labels.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

# Convert labels
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Build CNN model
model = Sequential()

model.add(Conv2D(
    32,
    (5,5),
    activation='relu',
    input_shape=X_train.shape[1:]
))

model.add(Conv2D(
    32,
    (5,5),
    activation='relu'
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Training Model...")

# Train model
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, y_test)
)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
model.save("model/traffic_model.h5")

print("Model saved successfully!")
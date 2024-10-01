from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from tensorflow.keras.layers import Dense, Flatten, Dropout  # Layers for the neural network
from tensorflow.keras.models import Sequential  # Sequential model class
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # Convolutional and pooling layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Preprocessing for MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # Image loading and conversion functions
from tensorflow.keras.utils import to_categorical  # To convert labels to categorical format
from sklearn.preprocessing import LabelBinarizer  # For label binarization
from sklearn.model_selection import train_test_split  # To split the dataset into training and testing sets
import numpy as np  
import os  
import cv2  # For image processing

# Directory containing the image data
DIRECTORY = "dataset"
# Categories of images to classify
CATEGORIES = ["cat", "dog"]

EPOCHS = 10  # Number of epochs for training
BS = 32  # Batch size

print('[INFO]: loading images')

data = []
labels = []

# Iterate through each category
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    # Iterate through each image in the category directory
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        
        # Filter out non-image files (e.g., .DS_Store)
        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Load and resize the image to the target size
                image = load_img(img_path, target_size=(224, 224))
                # Convert the image to a NumPy array
                image = img_to_array(image)
                # Preprocess the image for MobileNetV2
                image = preprocess_input(image)

                # Append the image data and corresponding label to the lists
                data.append(image)
                labels.append(category)
                # print(f"Loaded image: {img_path}")  # Print loaded image path
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        else:
            print(f"Skipped non-image file: {img_path}")

# One-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)  # Transform labels into binary format
labels = to_categorical(labels)  # Convert to categorical (one-hot) format

# Convert data lists to NumPy arrays for processing
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split the data into training and testing sets (80/20 split)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data augmentation to improve model generalization
aug = ImageDataGenerator(
    rotation_range=20,  # Randomly rotate images
    zoom_range=0.15,  # Randomly zoom into images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.15,  # Shear images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode="nearest"  # Fill in new pixels after transformations
)

# Construct a Convolutional Neural Network (CNN) model
model = Sequential()

# Add convolutional and pooling layers to the model
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))  # First convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # First max pooling layer

model.add(Conv2D(64, (3, 3), activation='relu'))  # Second convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Second max pooling layer

model.add(Conv2D(128, (3, 3), activation='relu'))  # Third convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Third max pooling layer

# Flatten the output from the previous layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # Fully connected layer
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
model.add(Dense(2, activation='softmax'))  # Output layer for 2 classes (cat and dog)

# Compile the model with optimizer, loss function, and metrics to track
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the shapes of training and testing data
print(f"trainX shape: {trainX.shape}")
print(f"trainY shape: {trainY.shape}")
print(f"testX shape: {testX.shape}")
print(f"testY shape: {testY.shape}")

# Train the model with training data and validate using testing data
model.fit(trainX, trainY, EPOCHS, BS, validation_data=(testX, testY))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(testX, testY)
print(f"test accuracy: {accuracy * 100:.2f}%")  # Print test accuracy

# Function to predict the category of an image
def predict_image(image_path):
    image = cv2.imread(image_path)
    # Resize the image to 224x224 and normalize pixel values
    image = cv2.resize(image, (224, 224)) / 255.0
    # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "cat" if np.argmax(prediction) == 0 else "dog"

# Predict the category for a sample image and print the result
result = predict_image("dataset/model_predict/14.jpg")
print(f"result {result}")  

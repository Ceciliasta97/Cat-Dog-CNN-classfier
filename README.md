# Cat - Dog Classification Project

## Overview
developing a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of cats and dogs. 


## Installation

1. Clone this repository to your build machine using:

```bash
  git clone https://github.com/Ceciliasta97/Cat-Dog-CNN-classfier.git
```
2. Navigate to the project directory:

```bash
  cd Cat-Dog-CNN-Classfier
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```

4. Run the file "Categorial.py"



## Key Components of the Project:
Image Preprocessing: using TensorFlow's ImageDataGenerator and Keras to loading and process images. (resizing the images to a uniform size of 224x224 pixels and normalizing pixel values to enhance model performance.)

Data Augmentation: prevent overfitting, includes random rotations, shifts, shearing, and horizontal flips of the images.

CNN Model Architecture: A sequential CNN architecture is constructed, consisting of multiple convolutional layers followed by max pooling layers. The model use ReLU activation function and concludes with a softmax layer to classify the images into two categories (cat and dog).

Training and Validation: The model is trained on a training dataset, with a split of 80% for training and 20% for testing. 

Prediction Functionality: A prediction function is included to classify new images. 



## Future Improvements
This project can be further enhanced by:

use a larger dataset for training to improve accuracy.
Implementing transfer learning techniques using pre-trained models.
Experimenting with different architectures to optimize performance.


## Contribution
Feel free to contribute and enhance the project!


## Contact
For any inquiries or issues, please contact cecilia at ceciliasta97@gmail.com

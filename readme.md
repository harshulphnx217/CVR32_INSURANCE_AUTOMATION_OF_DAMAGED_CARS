# **Car Damage Detection**
This repository contains code for estimating the severity of car damage based on input images. The code utilizes deep learning models to categorize the image, determine the location of the damage, and assess the severity of the damage.

##**Table of Contents**
* Introduction
* Setup
* Usage
* Requirements
* License 

##**Introduction**

The Car Damage Detection model aims to provide a system that can automatically evaluate the severity of car damage from images. It utilizes deep learning models trained on car images to perform various assessments, including categorizing the image as a car or not, determining the location of the damage, and estimating the severity of the damage.

The code consists of several components:

* First Gate: Determines if the image contains a car.
* Second Gate: Determines if the car is damaged.
* Location Assessment: Determines the location of the car damage (front, rear, or side).
* Severity Assessment: Estimates the severity of the car damage (minor, moderate, or severe).

## **Setup**
To set up the Car Damage Severity Estimation code, follow these steps:

1. Clone the repository:
```
git clone <repository-url>
cd car-damage-detection
```

2. Install the required dependencies. Please refer to the Requirements section for the list of dependencies.

3. Download the pre-trained models for each component:

* First Gate: Download the pre-trained weights for the VGG16 model trained on ImageNet.
* Second Gate: Download the pre-trained weights for the model used to determine car damage.
* Location Assessment: Download the pre-trained weights for the model used to determine the location of the damage.
* Severity Assessment: Download the pre-trained weights for the model used to estimate the severity of the damage.
4. Update the file paths in the code to point to the downloaded model weights and other necessary files.

##**Usage**
The Car Damage Detection model can be used to assess the severity of car damage from images. Follow these steps to use the code:

1. Ensure that you have set up the code following the Setup instructions.

2. Run the engine() function to start the assessment process. The function will prompt you to submit an image link or type 'exit' to quit.

3. The code will perform the following assessments:

* Validate if the image contains a car.
* Validate if the car is damaged.
* Determine the location of the car damage.
* Estimate the severity of the car damage.
4. The assessment results will be displayed on the console.

5. You can repeat the assessment process by running the engine() function again.

## **Requirements**
The Car Damage Detection model requires the following dependencies:

* Python (version >= 3.x)
* TensorFlow (version >= 2.x)
* Keras (version >= 2.x)
* NumPy
* Matplotlib
* PIL (Python Imaging Library)
* Requests (for downloading model weights)
Other dependencies as specified in the code

## **Model Architecture**
The model architecture is based on the VGG16 convolutional neural network. The pre-trained VGG16 model is used as a base model, which has learned to extract useful features from images. The top layers of the pre-trained model are modified and fine-tuned to adapt it to the specific classification task. The added layers include fully connected layers and softmax layers for predicting the damage location and severity.

## **Training**
The training process involves the following steps:

1. Dataset Preparation: The dataset is prepared by collecting car images and manually labeling them with their damage location and severity.

2. Data Augmentation: Data augmentation techniques, such as random cropping, flipping, and rotation, are applied to the training images to increase the model's ability to generalize.

3. Model Training: The model is trained using the training dataset. The training process involves adjusting the model's internal parameters (weights and biases) based on the input images and their known labels. The optimization algorithm used is stochastic gradient descent (SGD) with a learning rate schedule.

4. Evaluation: The trained model is evaluated using a separate validation dataset. Various performance metrics, such as accuracy, precision, recall, and F1-score, are calculated to assess the model's performance.

## **Code**
1. `get_predictions(preds, top=5)`: This function takes the predictions from a model and returns the top predictions along with their respective class labels and probabilities. It utilizes a pre-defined class index to map the class labels to human-readable names.

2. `prepare_img_224(img_path)`: This function prepares the image for input to the VGG16 model. It downloads the image from the provided  `img_path`, resizes it to a target size of 224x224 pixels, converts it to a NumPy array, adds an extra dimension for batch size, and applies preprocessing specific to the VGG16 model.

3. `car_categories_gate(img_224, model`): This function serves as the first gate in the assessment process. It takes the preprocessed image and a model (VGG16 in this case) as inputs. It uses the model to predict the class probabilities of the image and checks if any of the top predictions match the pre-defined car categories. If there is a match, it returns True indicating that the image contains a car, otherwise it returns False.

4. `prepare_img_256(img_path)`: This function prepares the image for input to the damage classification model. It downloads the image from the provided img_path, resizes it to a target size of 256x256 pixels, converts it to a NumPy array, and normalizes the pixel values.

5. `car_damage_gate(img_256, model)`: This function serves as the second gate in the assessment process. It takes the preprocessed image and a damage classification model as inputs. It uses the model to predict the probability of the car being damaged. If the predicted probability is below a threshold (0.5 in this case), it returns True indicating that the car is damaged, otherwise it returns False.

6. `location_assessment(img_256, model)`: This function determines the location of the car damage. It takes the preprocessed image and a location assessment model as inputs. It uses the model to predict the class probabilities of the image and selects the class label with the highest probability. The class labels correspond to different locations of the car damage (front, rear, or side). The function prints the assessment result.

7. `severity_assessment(img_256, model)`: This function estimates the severity of the car damage. It takes the preprocessed image and a severity assessment model as inputs. It uses the model to predict the class probabilities of the image and selects the class label with the highest probability. The class labels correspond to different levels of severity (minor, moderate, or severe). The function prints the assessment result.

8. `engine()`: This function serves as the main driver function for the car damage severity estimation process. It prompts the user to provide an image link or type 'exit' to quit. It calls the necessary functions in sequence to perform the assessment process, including validating if the image contains a car, if the car is damaged, determining the location of the damage, and estimating the severity of the damage. The function displays the assessment results on the console and allows the user to repeat the assessment process for multiple images.

These functions work together to assess the severity of car damage based on input images.

## **Results**
The model achieved promising results in classifying the damage location and severity of car images. Here are some key highlights:

1. Damage Location Classification: The model achieved an accuracy of 75% on the validation dataset for classifying the damage location into front, rear, and side.

2. Damage Severity Classification: The model achieved an accuracy of 80% on the validation dataset for classifying the damage severity into minor, moderate, and severe.

The model's performance was compared to an existing model, and it demonstrated superior accuracy and performance in both damage location and severity classification tasks.

## **Advantages of the Model**
This model offers several advantages over existing models for damage location and severity classification:

1. Improved Accuracy: The model achieved higher accuracy compared to existing models, indicating its ability to effectively distinguish between different damage locations and severity levels.

2. Fine-tuning Capability: The model leverages transfer learning by using a pre-trained VGG16 model as a base, enabling the adaptation of learned features to the specific task through fine-tuning.

3. Robustness: The model incorporates data augmentation techniques during training, making it more robust to variations in car image samples and increasing its generalization ability.

4. Easy Deployment: The trained model can be easily deployed in real-world applications to classify the damage location and severity of car images in an automated and efficient manner.

## **Conclusion**
The damage location and severity classification model presented in this project demonstrates improved accuracy and performance compared to existing models. By leveraging deep learning techniques, transfer learning, and data augmentation, the model effectively learns to classify car images according to their damage location and severity. This model can be valuable in various domains, such as insurance claims processing, automotive industry, and accident analysis, enabling automated and accurate damage assessment.

## **License**
This project is licensed under the MIT License.

Please note that the code provided is a sample and assumes the existence of certain files and data. Make sure to update the code with the correct file paths and data references according to your specific setup.

Feel free to modify and adapt the code to suit your needs.
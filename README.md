# mood-detection-using-handwriting-artificial-neural-networks-project

A CNN-based project that predicts a person’s mood from handwriting images using image processing and deep learning. Includes data augmentation, mood classification, and motivational message display based on the predicted emotion.
# Mood Detection Using Handwriting (CNN Based)

This project aims to predict a person's mood based on an image of their handwriting using a Convolutional Neural Network (CNN). It utilizes image preprocessing, data augmentation, and a trained CNN model to classify moods into categories such as Happy, Sad, Angry, etc.

## 🧠 Project Goal
To build a deep learning model that can accurately classify moods from handwriting images, deploy it through a web interface, and display a motivational message based on the predicted emotion.

## 🗂️ Dataset
- The dataset consists of handwriting samples labeled with moods.
- Created/simulated using text generation and image rendering tools (due to lack of public handwriting+mood datasets).
- Images are processed and augmented to increase model generalization.

## ⚙️ Technologies & Libraries
- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib, Seaborn
- Scikit-learn

## 🧪 Model
- A Convolutional Neural Network (CNN) was designed for image classification.
- Includes multiple Conv2D and MaxPooling layers.
- Trained on augmented handwriting images with mood labels.
- Achieves ~61% validation accuracy (can be improved with more data).

## 📈 Features
- Image preprocessing and grayscale conversion
- CNN training and evaluation with validation data
- Confusion matrix & performance visualization
- Displays motivational messages based on prediction
- Web interface for uploading handwriting images and displaying results

## 🚀 Future Improvements
- Use a larger and more diverse handwriting dataset
- Incorporate transfer learning (e.g., MobileNet, EfficientNet)
- Enhance UI with confidence score and real-time feedback
- Add multi-language handwriting support

## 🖼️ Output
- Displays uploaded handwriting
- Predicts and prints mood with confidence score
- Shows motivational message based on detected mood

## 🏁 How to Run
1. Clone the repo
2. Run the notebook `mood detection using handwriting.ipynb` in Google Colab or locally
3. Upload handwriting image
4. See predicted mood and message

## 🧑‍💻 Author
Developed as part of an AI mini project to explore emotion detection using vision-based deep learning.


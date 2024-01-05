# ASL_Letter_Detection
This application activates the user's webcam and uses the live feed to detect the ASL letters and digits being displayed.

train_model.py parses and normalizes the training and testing data and trains the model with it. It creates a .csv and .joblib file to be used by the other python script.
asl_detection_webcam.py initializes the user's webcam and uses MediaPipe's HandRecognizer module in conjunction with the trained model to detect ASL letters being displayed.

The model uses these datasets for the ASL Detection
- https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset (digits)
- https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset (letters)







# ASL_Letter_Detection
This application activates the user's webcam and uses the live feed to detect the ASL letters and digits being displayed.

train_model.py parses and normalizes the training and testing data and trains the model with it. It creates a .csv and .joblib file to be used by the other python script.
asl_detection_webcam.py initializes the user's webcam and uses MediaPipe's HandRecognizer module in conjunction with the trained model to detect ASL letters being displayed.

The model was trained with these datasets found on Kaggle: 
- https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset (digits)
- https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset (letters)


<img width="500" alt="Screen Shot 2024-01-05 at 5 54 41 PM" src="https://github.com/osher-steel/ASL_Letter_Detection/assets/111786194/bffaf5f7-1871-41ec-a3e1-d77c0143a636">

<img width="500" alt="Screen Shot 2024-01-05 at 5 56 49 PM" src="https://github.com/osher-steel/ASL_Letter_Detection/assets/111786194/a0a17040-b0c3-415d-8072-4d313c64f01b">





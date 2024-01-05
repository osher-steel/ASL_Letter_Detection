<img width="1440" alt="Screen Shot 2024-01-05 at 5 54 41 PM" src="https://github.com/osher-steel/ASL_Letter_Detection/assets/111786194/30bc4b9d-c433-40d7-a28a-fd9d9ce71c2a"># ASL_Letter_Detection
This application activates the user's webcam and uses the live feed to detect the ASL letters and digits being displayed.

train_model.py parses and normalizes the training and testing data and trains the model with it. It creates a .csv and .joblib file to be used by the other python script.
asl_detection_webcam.py initializes the user's webcam and uses MediaPipe's HandRecognizer module in conjunction with the trained model to detect ASL letters being displayed.

The model uses these datasets for the ASL Detection
- https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset (digits)
- https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset (letters)


Example:
<img width="1440" alt="Screen Shot 2024-01-05 at 5 54 41 PM" src="https://github.com/osher-steel/ASL_Letter_Detection/assets/111786194/adc2a7ba-c355-4ae2-ad48-0ec1a0955b97">

<img width="1440" alt="Screen Shot 2024-01-05 at 5 56 49 PM" src="https://github.com/osher-steel/ASL_Letter_Detection/assets/111786194/e49d2d9c-dab3-4b05-9cc9-edd573ce5bd8">




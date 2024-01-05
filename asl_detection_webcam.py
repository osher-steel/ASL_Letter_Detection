import cv2 as cv
from utils.visualizer import visualize_landmarks
from utils.HandRecognizer import HandRecognizer
import joblib
import pandas as pd
from train_model import normalize
import numpy as np
import contextlib
from io import StringIO

classifier = joblib.load('results/ASL_A2G_model.joblib')
classifier.n_jobs = 1

def model_prediction(recognizer_result,img):
    # Normalize recognizer result
    landmark = normalize(recognizer_result,img)

    #Turn into df
    df = pd.DataFrame([landmark], index=['Test'], columns=[f"Feature_{i}" for i in range(1, 43)])

    y_pred = classifier.predict(df)
    y_prob = classifier.predict_proba(df)

    #The highest probability is the one the model chooses as its prediction
    pred_prob =  np.max(y_prob[0])

    return y_pred, pred_prob

def main():
    #mhdc : minimum hand detection confidence
    #mhpc : minimum hand presence confidence
    hand_recognizer = HandRecognizer(mhdc=0.3,mhpc=0.3)

    # Start live stream from webcam
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Exiting")
            break

        #Flip frame on horizontal axis
        frame = cv.flip(frame,1)

        #Process frame for hand detection
        hand_recognizer.process(frame)

        # Accessing hand landmarks
        if hand_recognizer.result and hasattr(hand_recognizer.result, 'hand_landmarks'):
            hand_landmarks = hand_recognizer.result.hand_landmarks

            if hand_landmarks:
                # Generate prediction and probability with the model
                prediction, max_prob = model_prediction(hand_recognizer.result,frame)

                #Adds visuals to the image
                frame = visualize_landmarks(frame,hand_landmarks,prediction[0],max_prob)

        cv.imshow('ASL Letter Detection', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    with contextlib.redirect_stdout(StringIO()):
        main()
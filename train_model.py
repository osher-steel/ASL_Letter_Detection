import mediapipe as mp 
import numpy as np
import joblib
import os
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm

model_path = '/Users/oshersteel/Documents/Personal_Projects/Computer Vision/Models/gesture_recognizer.task'

training_set = '/Users/oshersteel/Documents/Personal_Projects/Computer Vision/ASL Detection/ASL_Alphabet_Dataset/asl_alphabet_train'
testing_set = '/Users/oshersteel/Documents/Personal_Projects/Computer Vision/ASL Detection/ASL_Alphabet_Dataset/asl_alphabet_test'

training_csv = 'results/landmark_training_set3.csv'
testing_csv = 'results/landmark_testing_set3.csv'

def create_hand_recognizer():
    # Create recognizer with Image mode
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    recognizer = GestureRecognizer.create_from_options(options)

    return recognizer

def countfiles():
    n = 0
    for folder in os.listdir(training_set):
        folderpath = os.path.join(training_set, folder)

        if os.path.isdir(folderpath):
            for file in os.listdir(folderpath):
                filepath = os.path.join(folderpath, file)

                if os.path.isfile(filepath):
                    n+=1
    
    for file in os.listdir(testing_set):
        filepath = os.path.join(testing_set,file)
        if os.path.isfile(filepath):
            n+=1

    return n

def normalize(recognition_result,img):
    width = 0
    height = 0

    if isinstance(img,np.ndarray):
        width= img.shape[0]
        height= img.shape[1]
    else:
        width = img.width
        height = img.height

    # Retrieve the landmark
    hand_landmark = recognition_result.hand_landmarks
    normalized = []
    
    if hand_landmark and hand_landmark[0]:
        # Convert the landmark to x,y coordinates
        converted_landmark = []
        for landmark in  hand_landmark[0]:
            point = (int(landmark.x * width),int(landmark.y * height))
            converted_landmark.append(point)

        min_x, min_y = np.min(converted_landmark, axis=0)
        max_x, max_y = np.max(converted_landmark, axis=0)

        width = max_x - min_x
        height = max_y - min_y

        # Normalize and flatten
        for point in converted_landmark:
            x = (point[0]-min_x)/width
            y = (point[1]-min_y)/height
            normalized.append(x)
            normalized.append(y)

    return normalized

def folder2landmark_set(folderpath, foldername, recognizer,pbar, is_training_data):
    landmark_set = []

    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)

        if not os.path.isfile(filepath):
            continue

        img = mp.Image.create_from_file(filepath)

        # Process images for hand landmarks
        recognition_result = recognizer.recognize(img)

        # Normalizes and flattens the landmark data into a 1d array
        normalized = normalize(recognition_result,img)
        
        #For training set the label is obtained from the folder name while the testing set has the label in the file name
        if is_training_data:
            label = foldername
        else:
            label = os.path.splitext(filename)[0]
        
        if normalized:
            normalized.append(label)
            landmark_set.append(normalized)
            pbar.update(1)

    return landmark_set

def create_csv(filepath,landmark_set):
    if os.path.exists(filepath):
        os.remove(filepath)

    with open(filepath,'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = [f"Feature_{i}" for i in range(1, 43)] + ["Letter"]
        csv_writer.writerow(header)
        csv_writer.writerows(landmark_set)

def prepare_data():
    recognizer = create_hand_recognizer()

    with tqdm(total=countfiles(), desc='Preprocessing Data') as pbar:
        training_root = sorted(os.listdir(training_set))
        landmark__training_set = []

        for foldername in training_root:
            folderpath = os.path.join(training_set, foldername)

            if os.path.isdir(folderpath):
                #Extend the training set with each new folder
                folder_set= folder2landmark_set(folderpath,foldername,recognizer,pbar, is_training_data=True)
    
                landmark__training_set.extend(folder_set)

        print('Done')
        # The testing set only has one folder
        landmark__testing_set = folder2landmark_set(testing_set,'null',recognizer,pbar, is_training_data=False)

        create_csv(training_csv,landmark__training_set)
        create_csv(testing_csv,landmark__testing_set)

def train_model():
    training_data = pd.read_csv(training_csv)
    testing_data = pd.read_csv(testing_csv)

    # Label is positioned at the last column
    X_train = training_data.iloc[:, :-1]  
    y_train = training_data.iloc[:, -1]   

    X_test = testing_data.iloc[:, :-1]  
    y_test = testing_data.iloc[:, -1]   

    # Train Model
    estimators = 100

    classifier = RandomForestClassifier(n_estimators=estimators, random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate Model
    y_preds = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)

    print(f'Accuracy: {accuracy}')

    # for pred,actual in zip(y_preds,y_test):
    #     print(f'Prediction :{pred} Actual:{actual}')

    joblib.dump(classifier, 'results/ASL_A2G_model.joblib')

def main():
    # prepare_data()
    train_model()

if __name__ == "__main__":
    main()


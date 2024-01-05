import cv2 as cv
import numpy as np

def convert_landmarks(image,landmarks):
    converted = []

    # Transform the landmarks to x,y coordinates based on the image dimensions
    for landmark in landmarks:
            image_height, image_width, _ = image.shape
            point = (int(landmark.x * image_width),int(landmark.y*image_height))
            converted.append(point)
    
    return converted

def draw_connections(coordinates,image,line_color,line_thickness):
    # Each coordinates 0-21 represents a point on the hand with 0 being the wrist

    fingers = []
    for i in range(5):
        start_point = (i * 4) +1
        finger = []

        for i in range(4):
            finger.append(start_point+i)
        
        fingers.append(finger)

    # Other landmark positions to be used for connections
    inner_connection = [5,9,13,17]
    wrist_connection = [1,5,17]

    # Connects all points in finger
    for finger in fingers:
        for i in range(0,3):
            cv.line(image,coordinates[finger[i]],coordinates[finger[i+1]],line_color,line_thickness)
    
    # Connects all knuckles except thumb
    for i in range(len(inner_connection)-1):
        cv.line(image,coordinates[inner_connection[i]],coordinates[inner_connection[i+1]],line_color,line_thickness)

    # Connects wrist with the knuckle of the thumb, index, and pinky
    for i in range(len(wrist_connection)):
        cv.line(image,coordinates[0],coordinates[wrist_connection[i]],line_color,line_thickness)

def visualize_landmarks(image,hand_landmarks, prediction, probability, point_color=(255, 0, 255),point_radius=10, line_color=(233,86,47), line_thickness=5):
    for hand_landmark in hand_landmarks:
        # Convert landmarks to coordinates
        converted_landmarks = convert_landmarks(image, hand_landmark)

        # Draws connections between landmarks
        draw_connections(converted_landmarks, image, line_color, line_thickness)

        # Draw circles ast each landmark coordinate
        for landmark in converted_landmarks:
            cv.circle(image, landmark , point_radius, point_color, -1)
    
        # Retrieve hand bounding coordinates to position text
        min_x, min_y = np.min(converted_landmarks, axis=0)
        max_x, max_y = np.max(converted_landmarks, axis=0)

        pred_position = (int(min_x + (max_x - min_x) / 2), int(min_y - 30))
        prob_position = (int(min_x + (max_x - min_x) / 2), int(max_y + 30))
        
        cv.putText(image, prediction, pred_position, cv.FONT_HERSHEY_TRIPLEX, 3.0, (255, 0, 0), 3)
        cv.putText(image, f'{probability}%', prob_position, cv.FONT_HERSHEY_TRIPLEX, 1.0, (230, 216, 173), 3)

    return image
import mediapipe as mp
import time

class HandRecognizer:
    def __init__(self, n_hands=1, mhdc=0.5, mhpc=0.5, mhtc=0.5):
        self.result = mp.tasks.vision.GestureRecognizerResult
        self.recognizer = mp.tasks.vision.GestureRecognizer
        self.create(n_hands, mhdc, mhpc, mhtc)
    
    def create(self, n_hands, mhdc, mhpc, mhtc):

        #Call back function to retrieve result from analyzing live frame
        def get_result(result:mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result
        
        model_path ='/Users/oshersteel/Documents/Personal_Projects/Computer Vision/Models/gesture_recognizer.task'

        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.LIVE_STREAM, num_hands=n_hands,
        min_hand_detection_confidence=mhdc,min_hand_presence_confidence=mhpc, min_tracking_confidence=mhtc, result_callback=get_result)
        
        self.recognizer = self.recognizer.create_from_options(options)
    
    # Processes the image for hand landmarks
    def process(self,frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.recognizer.recognize_async(mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        self.recognizer.close()
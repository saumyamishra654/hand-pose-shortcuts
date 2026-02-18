import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

#path to gesture recognition model
model_path = '/Users/saumyamishra/Desktop/Projects/hand-pose-shortcuts/gesture_recognizer.task'

#initialise video capture
cap = cv2.VideoCapture(1)

#written here by callback, and read by main loop
latest_result = None

#mp options
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# creating a gesture recognizer instance with the live stream mode
def print_result(result, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #image that cv2 creates needs to be convted to mp.Image media type
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        recognizer.recognize_async(mp_image, int(time.time()*1000))

        #read from last callback
        if latest_result and latest_result.gestures:
            gesture = latest_result.gestures[0][0]
            print(f"{gesture.category_name} ({gesture.score:.2f})")

        #break if you press q
        cv2.imshow('feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows
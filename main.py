import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import subprocess

gesture_actions = {
    'Pointing_Up': 'roshni',
}

last_gesture = None
gesture_frame_count = 0
last_triggered = 0

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
    global latest_result
    latest_result = result

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

        #read from last callback (main loop)
        if latest_result and latest_result.gestures and latest_result.hand_landmarks:
            #print(latest_result.hand_landmarks[0]) did this to see what hand_landmarks looks like
            #we make the bounding box by taking the minimum and maximum x and y coordinates of the hand landmarks, and then drawing a rectangle around it
            x = []
            y = []
            for landmark in latest_result.hand_landmarks[0]:
                x.append(landmark.x)
                y.append(landmark.y)
            x_min = int(min(x) * frame.shape[1])
            x_max = int(max(x) * frame.shape[1])
            y_min = int(min(y) * frame.shape[0])
            y_max = int(max(y) * frame.shape[0])
            if latest_result.gestures[0][0].category_name != last_gesture:
                gesture_frame_count = 0
                last_triggered = 0
            if gesture_frame_count > 5 and (time.time() - last_triggered) > 2:
                action = gesture_actions.get(latest_result.gestures[0][0].category_name)
                if action:
                    print(f"Performing action: {action}")
                    subprocess.run(['shortcuts', 'run', action])
                    last_triggered = time.time()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            gesture = latest_result.gestures[0][0]
            cv2.putText(frame, gesture.category_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"{gesture.category_name} ({gesture.score:.2f})")
            last_gesture = gesture.category_name
            gesture_frame_count += 1

        #break if you press q
        cv2.imshow('feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#release camera and destroy windows
cap.release()
cv2.destroyAllWindows()
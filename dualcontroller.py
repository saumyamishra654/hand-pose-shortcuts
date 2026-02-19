# in this program, we will use the mediapipe hand gesture recognition
# to make a two hand controller that can be used to control system functions like 
# volume, brightness etc

#will begin by importing things
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import subprocess
import numpy as np

last_distance = 0
current_volume = 50
last_volume_update = 0
current_brightness = 0.5

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
    num_hands=2,
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
        if latest_result and latest_result.gestures and latest_result.hand_landmarks and latest_result.handedness:
            #print(latest_result) #did this to see what hand_landmarks looks like
            #we make the bounding box by taking the minimum and maximum x and y coordinates of the hand landmarks, and then drawing a rectangle around it
            hands = {}
            for i, lms in enumerate(latest_result.hand_landmarks):
                label = latest_result.handedness[i][0].category_name
                hands[label] = {
                    'landmarks': lms,
                    'gesture': latest_result.gestures[i][0]
                }
            
            left = hands.get('Left')
            right = hands.get('Right')
            #need to make separate bounding boxes for each hand, read action from left hand and then later implement distance mapping logic
            if left:
                #bounding box for left hand
                x = []
                y = []
                for landmark in left['landmarks']:
                    x.append(landmark.x)
                    y.append(landmark.y)
                x_min = int(min(x) * frame.shape[1])
                x_max = int(max(x) * frame.shape[1])
                y_min = int(min(y) * frame.shape[0])
                y_max = int(max(y) * frame.shape[0])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

                #this hand controls the choice of function, so we read the gesture from here
                gesture = left['gesture']
                cv2.putText(frame, gesture.category_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if right:
                #for this, hand, we need to implement the distance mapping logic. instead of bounding box, we will draw a line showing the distance between the two fingers
                thumb_tip = right['landmarks'][4]
                index_tip = right['landmarks'][8]

                x1 = int(thumb_tip.x * frame.shape[1])
                y1 = int(thumb_tip.y * frame.shape[0])
                x2 = int(index_tip.x * frame.shape[1])
                y2 = int(index_tip.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                #calculate distance between thumb tip and index tip
                distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
                cv2.putText(frame, f"d={distance:.3f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if left:
                    if last_distance is not None:
                        delta = distance - last_distance
                        if abs(delta) > 0.01 and time.time() - last_volume_update > 0.5:
                            left_gesture = left['gesture'].category_name
                            if left_gesture == 'Open_Palm':  # volume mode
                                current_volume = max(0, min(100, current_volume + int(delta * 500)))
                                subprocess.Popen(['osascript', '-e', f'set volume output volume {current_volume}'])
                                last_volume_update = time.time()
                                print(f"Volume: {current_volume}")
                            elif left_gesture == 'Closed_Fist':  # brightness mode
                                current_brightness = max(0.0, min(1.0, current_brightness + delta * 5))
                                subprocess.Popen(['brightness', f'{current_brightness:.2f}'])
                                last_volume_update = time.time()
                                print(f"Brightness: {current_brightness:.2f}")

                last_distance = distance  # always update


        #break if you press q
        cv2.imshow('feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#release camera and destroy windows
cap.release()
cv2.destroyAllWindows()


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
import webbrowser
import os

last_distance = None
smoothed_distance = None
current_volume = 50
last_volume_update = 0
current_brightness = 0.5
brightness_backend = 'brightness'
reported_brightness_unverified = False
reported_brightness_blocked = False

PINCH_MIN = 0.03
PINCH_MAX = 0.25
DISTANCE_SMOOTHING_ALPHA = 0.35
CONTROL_UPDATE_INTERVAL = 0.15
YOUTUBE_URL = 'https://youtu.be/92ydUdqWE1g?si=edrLceeo2uEqSdZr/'
YOUTUBE_TRIGGER_COOLDOWN = 4.0
FOCUS_CHECK_INTERVAL = 0.5
CAMERA_INDEX = 1

last_youtube_trigger = 0.0
both_pointing_up_active = False


def set_system_volume(volume: int) -> int | None:
    script_set = f"set volume output volume {volume} without output muted"
    script_get = "output volume of (get volume settings)"
    result = subprocess.run(
        ['osascript', '-e', script_set, '-e', script_get],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Volume update failed: {result.stderr.strip()}")
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        print(f"Unexpected volume response: {result.stdout.strip()}")
        return None


def map_distance_to_percent(distance: float) -> int:
    clamped = max(PINCH_MIN, min(PINCH_MAX, distance))
    normalized = (clamped - PINCH_MIN) / (PINCH_MAX - PINCH_MIN)
    return int(round(normalized * 100))


def set_brightness_with_keys(target_brightness: float, current_estimate: float) -> tuple[bool, float, bool]:
    global brightness_backend, reported_brightness_blocked

    if abs(target_brightness - current_estimate) < 0.03:
        return True, current_estimate, False

    step_size = 0.0625
    steps = max(1, int(round(abs(target_brightness - current_estimate) / step_size)))
    key_code = 144 if target_brightness > current_estimate else 145

    script = (
        'tell application "System Events"\n'
        f'  repeat {steps} times\n'
        f'    key code {key_code}\n'
        '  end repeat\n'
        'end tell'
    )

    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    if result.returncode != 0:
        error_text = (result.stderr or '').strip()
        if error_text:
            print(f"Brightness key fallback failed: {error_text}")
        lowered_error = error_text.lower()
        if (
            'not allowed assistive access' in lowered_error
            or 'not authorised' in lowered_error
            or 'not allowed to send keystrokes' in lowered_error
        ):
            brightness_backend = 'blocked'
            if not reported_brightness_blocked:
                print('Grant Accessibility permission to Terminal/VS Code to send brightness key events.')
                print('System Settings → Privacy & Security → Accessibility, then enable your terminal and VS Code.')
                reported_brightness_blocked = True
        return False, current_estimate, False

    return True, target_brightness, True


def set_system_brightness(target_brightness: float, current_estimate: float) -> tuple[bool, float, bool]:
    global brightness_backend

    if brightness_backend == 'blocked':
        return False, current_estimate, False

    if brightness_backend == 'brightness':
        result = subprocess.run(
            ['brightness', f'{target_brightness:.2f}'],
            capture_output=True,
            text=True
        )
        tool_output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        tool_failed = (
            result.returncode != 0
            or 'failed to set brightness' in tool_output
            or 'failed to get brightness' in tool_output
            or 'error -' in tool_output
        )
        if not tool_failed:
            return True, target_brightness, False

        stderr_text = (result.stderr or '').strip()
        if stderr_text:
            print(stderr_text)
        print('Brightness utility failed; switching to keyboard brightness fallback.')
        brightness_backend = 'keys'

    return set_brightness_with_keys(target_brightness, current_estimate)


def is_app_focused() -> bool:
    script = 'tell application "System Events" to unix id of first application process whose frontmost is true'
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    if result.returncode != 0:
        return True
    try:
        frontmost_pid = int(result.stdout.strip())
    except ValueError:
        return True
    return frontmost_pid == os.getpid()

#path to gesture recognition model
model_path = '/Users/saumyamishra/Desktop/Projects/hand-pose-shortcuts/gesture_recognizer.task'

#initialise video capture
cap = None
camera_active = False
window_focused = True
last_focus_check = 0.0
paused_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(paused_frame, 'Paused (window not focused)', (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

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
        now = time.time()
        if now - last_focus_check >= FOCUS_CHECK_INTERVAL:
            window_focused = is_app_focused()
            last_focus_check = now

        if window_focused and not camera_active:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                camera_active = True
                print('Camera resumed.')
                last_distance = None
                smoothed_distance = None
            else:
                if cap is not None:
                    cap.release()
                cap = None
                frame = paused_frame.copy()
                cv2.putText(frame, 'Waiting for camera...', (170, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.imshow('feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        if (not window_focused) and camera_active:
            if cap is not None:
                cap.release()
            cap = None
            camera_active = False
            last_distance = None
            smoothed_distance = None
            print('Camera paused (window not focused).')

        if not camera_active:
            frame = paused_frame.copy()
            cv2.imshow('feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if cap is None:
            continue

        ret, frame = cap.read()
        if not ret:
            if cap is not None:
                cap.release()
            cap = None
            camera_active = False
            continue

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

            if left and right:
                left_gesture_name = left['gesture'].category_name
                right_gesture_name = right['gesture'].category_name
                both_pointing_up = left_gesture_name == 'Pointing_Up' and right_gesture_name == 'Pointing_Up'
                now = time.time()
                if both_pointing_up:
                    if (not both_pointing_up_active) and (now - last_youtube_trigger > YOUTUBE_TRIGGER_COOLDOWN):
                        webbrowser.open(YOUTUBE_URL)
                        last_youtube_trigger = now
                        print(f'Opened YouTube: {YOUTUBE_URL}')
                    both_pointing_up_active = True
                else:
                    both_pointing_up_active = False
            else:
                both_pointing_up_active = False

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

                if smoothed_distance is None:
                    smoothed_distance = distance
                else:
                    smoothed_distance = ((1 - DISTANCE_SMOOTHING_ALPHA) * smoothed_distance) + (DISTANCE_SMOOTHING_ALPHA * distance)

                if left:
                    if last_distance is not None:
                        if time.time() - last_volume_update > CONTROL_UPDATE_INTERVAL:
                            left_gesture = left['gesture'].category_name
                            target_percent = map_distance_to_percent(smoothed_distance)
                            if left_gesture == 'Open_Palm':  # volume mode
                                if abs(target_percent - current_volume) >= 2:
                                    applied_volume = set_system_volume(target_percent)
                                else:
                                    applied_volume = None
                                last_volume_update = time.time()
                                if applied_volume is not None:
                                    current_volume = applied_volume
                                    print(f"Volume: {applied_volume}")
                            elif left_gesture == 'Pointing_Up':  # brightness mode
                                target_brightness = target_percent / 100.0
                                if abs(target_brightness - current_brightness) >= 0.04:
                                    brightness_ok, applied_brightness, unverified = set_system_brightness(target_brightness, current_brightness)
                                    if brightness_ok:
                                        current_brightness = applied_brightness
                                        if unverified:
                                            if not reported_brightness_unverified:
                                                print('Brightness key events sent; macOS does not provide programmatic readback here.')
                                                reported_brightness_unverified = True
                                            print(f"Brightness (target): {current_brightness:.2f}")
                                        else:
                                            print(f"Brightness: {current_brightness:.2f}")
                                last_volume_update = time.time()

                last_distance = distance  # always update


        #break if you press q
        cv2.imshow('feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#release camera and destroy windows
if cap is not None:
    cap.release()
cv2.destroyAllWindows()


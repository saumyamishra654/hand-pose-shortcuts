import cv2
import os
import datetime

cap = cv2.VideoCapture(0)
    
ret, frame = cap.read()
#save to directory called images, if it doesn't exist create it
if not os.path.exists('images'):
    os.makedirs('images')

if ret:
    timestamp = int(datetime.datetime.now().timestamp())
    filename = f'images/captured_image_{timestamp}.jpg'
    cv2.imwrite(filename, frame)
    print(f"image captured, {frame.shape}")
else:
    print("oppsie daisie")

cap.release()
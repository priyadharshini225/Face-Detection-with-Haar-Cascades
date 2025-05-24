# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

### PROGRAM
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline

model = cv2.imread('WhatsApp Image 2024-01-09 at 21.26.03_7aea8f7c.jpg',0)
withglass = cv2.imread('WITHGLASS.png',0)
group = cv2.imread('GROUP.png',0)

plt.imshow(model,cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/d9912b35-a432-4a8e-965d-5d852bde4d63)

```
plt.imshow(withglass,cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/cd5bc8d2-fb74-4c84-bf71-d0cd3d24a6e4)

```
plt.imshow(group,cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/46ef0e25-47e0-4573-9977-4ea9703a0c45)

```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_face(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 2) 
        
    return face_img
result = detect_face(withglass)


plt.imshow(result,cmap='gray')
plt.show()

```
![image](https://github.com/user-attachments/assets/2c04e095-8a7a-4557-9067-4aa162ca672b)

```
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/422acc22-215e-4578-9714-e88bc9cbf769)

```


def detect_face(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 4) 
        
    return face_img
    

# Doesn't detect the side face.
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/acbc2fc8-5a65-4f58-844f-9d61f41ff4ac)

```

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 4) 
        
    return face_img
    

result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/f69ebc12-219f-4343-a896-20a398d5e88a)


```

eyes = eye_cascade.detectMultiScale(withglass)
# White around the pupils is not distinct enough to detect Denis' eyes here!
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()
```
![image](https://github.com/user-attachments/assets/49e520f2-c58b-4832-b55b-e46a76c36f5c)


```


cap = cv2.VideoCapture(0)

# Set up matplotlib
plt.ion()
fig, ax = plt.subplots()

ret, frame = cap.read(0)
frame = detect_face(frame)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Video Face Detection')

while True:
    ret, frame = cap.read(0)

    frame = detect_face(frame)

    # Update matplotlib image
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.10)

   

cap.release()
plt.close()

```
![image](https://github.com/user-attachments/assets/d653e3e4-7f84-4949-8054-e3f17dcfed6b)




## RESULT:

Hence the program is completed successfully.

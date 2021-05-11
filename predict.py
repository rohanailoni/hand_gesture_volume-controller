import numpy as np
from tensorflow.keras.models import model_from_json
import operator
import cv2
import sys, os
import pyautogui
# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}
loop=1
while True:
    print("loop started:-",loop)
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (64, 64)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    #test_image=cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    # if np.argmax(result)==5 and loop%2==0:
    #     myScreenshot = pyautogui.screenshot()
    #     import random
    #     myScreenshot.save(r'C:\Users\91800\OneDrive - vit.ac.in\Desktop\ml\number-sign-recognition\screen\{}.png'.format(random.randint(1,10000)))
    # if np.argmax(result)==0:
    #     pyautogui.press('volumemute')
    # elif np.argmax(result)==5:
    #     pyautogui.press('volumeup')
    # elif np.argmax(result)==2:
    #     pyautogui.press('volumedown')
    # elif np.argmax(result)==4:
    #     pyautogui.press('volumeup')
    print(result)
    prediction = {'ZERO': result[0][0], 
                  'ONE': result[0][1], 
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5]}
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    loop+=1
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()

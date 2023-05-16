from imutils.video import VideoStream
import cv2
import time
import f_detector
import imutils
import numpy as np
import keyboard
from imutils.video import FileVideoStream
import os
import re

def read_blinks(fpath,max_frames=533):
    detector = f_detector.eye_blink_detector()
    COUNTER = 0
    TOTAL = 0
    #vs = VideoStream(src=0).start()
    #vs = FileVideoStream(path=fpath).start()
    cam=cv2.VideoCapture(fpath,apiPreference=cv2.CAP_FFMPEG)
    currentframe=1
    baseline=0
    start_time = time.time()
    tblinks=[]
    ablinks=[]
    while True:
        ret,im = cam.read()
        if currentframe >= max_frames or not ret: break
        #if not ret: break
        #im = cv2.flip(im, 1)
        im = imutils.resize(im, width=720)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rectangles = detector.detector_faces(gray, 0)
        boxes_face = f_detector.convert_rectangles2array(rectangles,im)
        if len(boxes_face)!=0:
            areas = f_detector.get_areas(boxes_face)
            index = np.argmax(areas)
            rectangles = rectangles[index]
            boxes_face = np.expand_dims(boxes_face[index],axis=0)
            COUNTER,TOTAL = detector.eye_blink(gray,rectangles,COUNTER,TOTAL)
            img_post = f_detector.bounding_box(im,boxes_face,['blinks: {}'.format(TOTAL)])
        else:
            img_post = im 
        end_time = (time.time() - start_time)/60
        baseline = TOTAL/currentframe
        tblinks.append(TOTAL)
        ablinks.append(baseline)
        cv2.putText(img_post,f"baseline blinks: {baseline}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.putText(img_post,f"Frames: {currentframe}",(10,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('blink_detection',img_post)
        currentframe+=1
        if cv2.waitKey(1) &0xFF == ord('q'):
            print(len(tblinks))
            print(tblinks)
            print(len(ablinks))
            print(ablinks)
            break
        
    while True:
        if currentframe >= max_frames: break
        currentframe+=1
        tblinks.append(None)
        ablinks.append(None)

    print(filename)
    print(len(tblinks))
    print(len(ablinks))
    average=np.array(ablinks)
    total=np.array(tblinks)
    cam.release()
    cv2.destroyAllWindows()
    del cam
    return total,average
    

def saveData(t, a):
  total = np.asarray(t)
  average = np.asarray(a)
  np.save('total_test_var', total)
  np.save('average_test_var', average)
  #print(f'Saving X and y of shape {X.shape}, {y.shape}, respectively.')

def savelabels():
    l=[]
    for i in range(61):
        l.append(0)
    for i in range(60):
        l.append(0)
    print(l)

path='videos'
t=[]
a=[]
l=[]
regexp1 = re.compile(r'l[1-7]')
regexp2 = re.compile(r't[1-7]')
regexp3 = re.compile(r'lie')
regexp4 = re.compile(r'truth')
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    if os.path.isfile(file_path):
        tempt,tempa=read_blinks(file_path)
        print(tempa)
        #if (not all(v == 0 for v in tempa)and not all(v == 0 for v in tempt)):
        t.append(tempt)
        a.append(tempa)
        if regexp1.search(file_path) or regexp3.search(file_path):
            l.append(0)
        if regexp4.search(file_path) or regexp2.search(file_path):
            l.append(1)

saveData(t,a)
labels = np.asarray(l)
np.save('labels_test', labels)
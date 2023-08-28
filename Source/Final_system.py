from itertools import count
import multiprocessing
from re import U
from threading import currentThread
from tkinter import Frame
import mediapipe as mp
import time
import cv2, queue, threading, time
import numpy as np
import tflite_runtime.interpreter as tflite
import board
import neopixel
import math
import colorsys
from os import walk
import random
import subprocess
from mutagen.mp3 import MP3
import threading
import alsaaudio
from queue import Queue

# version 1.0
# New music script
# version 1.1
# update neutral case
import keyboard
#120 led
#Run this file with command sudo -E python3 System.py
######################## Global setting ###########################################
MAX_LED = 120
global gLED_STATE #wait or working
gLED_STATE = 'wait'
# System State 0: idle || 1:working
global SYSTEM_STATE
global mp3_length
global max_result
SYSTEM_STATE = 0
max_result = 999
mp3_length =0
q = Queue() # for thread communication
currentEmotion = 'Neutral'
idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
# load model
interpreter = tflite.Interpreter(model_path="./emotions_mobilenet.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])
#Init light
pixels = neopixel.NeoPixel(board.D18, MAX_LED) # Raspberry Pi wiring!
LIGHTBLUE = []
YELLOW = []
ORANGE = []
RED = []
up_step = 1
down_step = -1
#init sound
PATH1_HAPPY ='./Music/Happy/'
PATH2_NEUTRAL ='./Music/Neutral/'
PATH3_DEFAULT = './Music/Default'
# PATH3 ='./MER_Audio/Q3'
# PATH4 ='./MER_Audio/Q4'
#get path of mp3 files
L1_HAPPY = []
L2_NEUTRAL = []
L3_DEFAULT = []
# L3 = []
# L4 = []
for (dirpath, dirnames, filenames) in walk(PATH1_HAPPY):
    L1_HAPPY.extend(filenames)
    break
for (dirpath, dirnames, filenames) in walk(PATH2_NEUTRAL):
    L2_NEUTRAL.extend(filenames)
    break
for (dirpath, dirnames, filenames) in walk(PATH3_DEFAULT):
    L3_DEFAULT.extend(filenames)
    break  
#############################################################################################################
############################# Platform ##########################
class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
 
        self.minDetectionCon = minDetectionCon
 
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
 
    def findFaces(self, img, draw=True):
 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img, max = self.fancyDraw(img,bbox)
                    global max_result
                    max_result = max
                    print("Global max_result" + str(max_result))
                    img = cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
                    img = cv2.putText(img, str(idx_to_class[max]), (bbox[0], bbox[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        else:
            print("No face")
        #return img, bboxs, max_result
        return img, bboxs
 
    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        crop_image = img[y:y1, x:x1]
        img_resized = cv2.resize(src=crop_image, dsize=(224, 224))
        input_image = img_resized.astype(np.float32)
        input_image = np.expand_dims(input_image, 0)
        interpreter.set_tensor(input_details[0]['index'], input_image)  
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index']) 
        max = np.argmax(output_data)
        if  0<= max or max <= 6:
          pass
        else:
          max =999
          print('N/A face')        

        return img, max

# hue from 0 to 65534, sat 0->255; vaule 0->255
def colorHSV(hue, sat, val):
        if hue >= 65536:
            hue %= 65536

        hue = (hue * 1530 + 32768) // 65536
        if hue < 510:
            b = 0
            if hue < 255:
                r = 255
                g = hue
            else:
                r = 510 - hue
                g = 255
        elif hue < 1020:
            r = 0
            if hue < 765:
                g = 255
                b = hue - 510
            else:
                g = 1020 - hue
                b = 255
        elif hue < 1530:
            g = 0
            if hue < 1275:
                r = hue - 1020
                b = 255
            else:
                r = 255
                b = 1530 - hue
        else:
            r = 255
            g = 0
            b = 0

        v1 = 1 + val
        s1 = 1 + sat
        s2 = 255 - sat

        r = ((((r * s1) >> 8) + s2) * v1) >> 8
        g = ((((g * s1) >> 8) + s2) * v1) >> 8
        b = ((((b * s1) >> 8) + s2) * v1) >> 8

        return r, g, b

def loop_script_caculate_rgb(up_step, down_step):
  light_blue = 35443 #light blue
  yellow = 10922 #yellow
  orange = 7098 # Orange
  red =  0 # red
  #BLUE
  cout = 0
  for i in np.arange(10,255,up_step):
    r,g,b =colorHSV(light_blue,255,i)
    light = [r,g,b]
    if (i == 254):
      print('Max light blue:')
      print(light)
      print('Index: ' + str(cout))
    cout = cout +1
    LIGHTBLUE.append(light)
  for i in np.arange(255,30,down_step):
    r,g,b =colorHSV(light_blue,255,i)
    light = [r,g,b]
    LIGHTBLUE.append(light)

#YELLOW
  for i in np.arange(10,255,up_step):
    r,g,b =colorHSV(yellow,255,i)
    light = [r,g,b]
    YELLOW.append(light)
  for i in np.arange(255,30,down_step):
    r,g,b =colorHSV(yellow,255,i)
    light = [r,g,b]
    YELLOW.append(light)

#ORANGE
  for i in np.arange(10,255,up_step):
    r,g,b =colorHSV(orange,255,i)
    light = [r,g,b]
    ORANGE.append(light)
  for i in np.arange(255,30,down_step):
    r,g,b =colorHSV(orange,255,i)
    light = [r,g,b]
    ORANGE.append(light)
  
  #RED
  for i in np.arange(10,255,up_step):
    r,g,b =colorHSV(red,255,i)
    light = [r,g,b]
    RED.append(light)
  for i in np.arange(255,30,down_step):
    r,g,b =colorHSV(red,255,i)
    light = [r,g,b]
    RED.append(light)

    
def turn_off_led():
      for i in range(MAX_LED):
        pixels[i] = (abs(0), abs(0), abs(0))

def display_led(color,start_element, stop_element):

    for j in range(start_element, stop_element):
        if color == 0:
            pixels.fill((abs(LIGHTBLUE[j][0]), LIGHTBLUE[j][1], LIGHTBLUE[j][2]))
        elif color == 1:
            pixels.fill((abs(YELLOW[j][0]), YELLOW[j][1], YELLOW[j][2]))
        elif color == 2:
            pixels.fill((abs(ORANGE[j][0]), ORANGE[j][1], ORANGE[j][2]))    
        elif color == 3:
            pixels.fill((abs(RED[j][0]), RED[j][1], RED[j][2])) 
        else:
            print('Error color')
            break

def wait(wait_time):
  start_time = time.time()
  while (time.time() - start_time) < wait_time:
    pass
def display_default_color():
  pixels.fill(255,255,255) #white

# def led_control(in_q):
    #extract transfered data
    # 0: waiting | 1: working
    # LED_STATE = 0
    # print('START LED CONTROL PROCESS')
    # while True:
    #     # Get data in
    #     #print("Getting data")
    #     if in_q.empty():
    #         #print('Led control thread - empty queue')
    #         LED_STATE = LED_STATE
    #     else:    
    #         data = in_q.get()

    #     #print("Data light control:" + str(data))
    #     # Handle thread state
    #     #{0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
    #     if data == 'Activate_unpleasant' or data == 'Unactivate_unpleasant' or data == 'Activated_pleasant' or data == 'Neutral':
    #         LED_STATE = 1
    #         print('Light control thread - Working mode')
    #     elif data == 'wait':
    #         LED_STATE = 0
    #         #print('Light control thread - Waiting mode')
    #     else:
    #         print('Default data handle case')
    #         print('LED STATE: ' + str(LED_STATE))
    #         pass
            
    #     # State handle
    #     if LED_STATE == 0:
    #         # waiting mode
    #         #time.sleep(0.2)
    #         # print("Light control -waiting mode")
    #         # print(type(data))
    #         # print(data)
    #         continue 
    #     else:
    #             print("Light control- start working")
    #             if data == 'Activate_unpleasant':
    #                 #Blue light
    #                 print("Light control 1")
    #                 display_led(0,0,244)
    #                 wait_time = 14
    #                 start_time = time.time()
    #                 while (time.time() - start_time) < wait_time:
    #                     if in_q.get() == 'wait':
    #                         LED_STATE = 0
    #                         print("Light control thread - Breaked light control")
    #                         break
    #                     else:
    #                         #do nothing
    #                         pass
    #                 display_led(0,245,len(LIGHTBLUE))
    #                 if LED_STATE == 0:
    #                     break
    #                 #Yellow light

    #                 display_led(1,0,244)
    #                 while (time.time() - start_time) < wait_time:
    #                     if in_q.get() == 'wait':
    #                         LED_STATE = 0
    #                         print("Light control thread - Breaked light control")
    #                         break
    #                     else:
    #                         #do nothing
    #                         pass
    #                 display_led(1,245,len(YELLOW))
    #                 LED_STATE = 0

    #             elif data == 'Unactivate_unpleasant':
    #                 # Yellow
    #                 print("Light control 2")                    
    #                 display_led(1,0,244)
    #                 wait_time = 14
    #                 start_time = time.time()
    #                 while (time.time() - start_time) < wait_time:
    #                     if in_q.get() == 'wait':
    #                         LED_STATE = 0
    #                         print("Light control thread - Breaked light control")
    #                         break
    #                     else:
    #                         # do nothing
    #                         pass
    #                 display_led(1,245,len(YELLOW))
    #                 # Orange
    #                 if LED_STATE == 0:
    #                     break
    #                 display_led(2,0,244)
    #                 wait_time = 14
    #                 start_time = time.time()
    #                 while (time.time() - start_time) < wait_time:
    #                     if in_q.get() == 'wait':
    #                         LED_STATE = 0
    #                         print("Light control thread - Breaked light control")
    #                         break
    #                     else:
    #                         pass
    #                 display_led(2,245,len(ORANGE))
    #                 LED_STATE = 0

    #             elif data == 'Activated_pleasant':
    #                 print("Light control 3")                    
    #                 display_led(0,0,244)
    #                 wait_time = 14
    #                 start_time = time.time()
    #                 while (time.time() - start_time) < wait_time:
    #                     if in_q.get() == 'wait':
    #                         LED_STATE = 0
    #                         print("Light control thread - Breaked light control")
    #                         break
    #                     else:
    #                         # do nothing
    #                         pass
    #                 display_led(0,245,len(LIGHTBLUE))
    #                 if LED_STATE == 0:
    #                     break
    #                 display_led(1,0,244)
    #                 wait_time = 14
    #                 start_time = time.time()
    #                 while (time.time() - start_time) < wait_time:
    #                     if in_q.get() == 'wait':
    #                         LED_STATE = 0
    #                         print("Light control thread - Breaked light control")
    #                         break
    #                     else:
    #                         # do thing
    #                         pass
    #                 display_led(1,245,len(YELLOW))
    #                 LED_STATE = 0                    
    #             else:
    #                 #Netral light
    #                 print("Light control 4")
    #                 display_led(2,0,244)
    #                 wait_time = 14
    #                 start_time = time.time()
    #                 while (time.time() - start_time) < wait_time:
    #                     if in_q.get() == 'wait':
    #                         LED_STATE = 0
    #                         print("Light control thread - Breaked light control")
    #                         break
    #                     else:
    #                         #do nothing
    #                         pass
    #                 display_led(2,245,len(ORANGE))
    #                 if LED_STATE == 0:
    #                     break
    #                 display_led(3,0,244)
    #                 wait_time = 14
    #                 start_time = time.time()
    #                 while (time.time() - start_time) < wait_time:
    #                     if in_q.get() == 'wait':
    #                         LED_STATE = 0
    #                         print("Light control thread - Breaked light control")
    #                         break
    #                     else:
    #                         #do nothing
    #                         pass
    #                 display_led(3,245,len(RED))
    #                 LED_STATE = 0                    

def light_process(event, state):
                #IslightControlStart = False
                print("INFO: Light process - start")
                if state == 'Activate_unpleasant':
                        #Blue light
                        print("Light control 1")
                        display_led(0,0,244)
                        wait_time = 10
                        start_time = time.time()
                        while (time.time() - start_time) < wait_time and event.is_set():
                                pass
                        display_led(0,245,len(LIGHTBLUE))
                        if not event.is_set():
                            return 0
                        display_led(1,0,244)
                        wait_time = 10
                        start_time = time.time()
                        while (time.time() - start_time) < wait_time and event.is_set():
                                pass
                        display_led(1,245,len(YELLOW))
                        return 1

                elif state == 'Unactivate_unpleasant':
                    # Yellow
                    print("Light control 2")                    
                    display_led(1,0,244)
                    wait_time = 10
                    start_time = time.time()
                    while (time.time() - start_time) < wait_time and event.is_set():
                            pass
                    display_led(1,245,len(YELLOW))

                    # Orange
                    if not event.is_set():
                        return 0
                    display_led(2,0,244)
                    wait_time = 10
                    start_time = time.time()
                    while (time.time() - start_time) < wait_time and event.is_set():
                            pass
                    display_led(2,245,len(ORANGE))
                    return 1
                    
                elif state == 'Activated_pleasant':
                    #Light blue
                    print("Light control 3")                    
                    display_led(0,0,244)
                    wait_time = 10
                    start_time = time.time()
                    while (time.time() - start_time) < wait_time and event.is_set():
                            pass
                    display_led(0,245,len(LIGHTBLUE))

                    #Yellow
                    if not event.is_set():
                        return 0
                    display_led(1,0,244)
                    wait_time = 10
                    start_time = time.time()
                    while (time.time() - start_time) < wait_time and event.is_set():
                            pass
                    display_led(1,245,len(YELLOW))                   
                else:
                    #Netral light
                    print("Light control 4")
                    display_led(2,0,244)
                    wait_time = 10
                    start_time = time.time()
                    while (time.time() - start_time) < wait_time and event.is_set():
                            pass
                    display_led(2,245,len(ORANGE))

                    if not event.is_set():
                        return 
                    display_led(3,0,244)
                    wait_time = 10
                    start_time = time.time()
                    while (time.time() - start_time) < wait_time and event.is_set():
                            pass
                    display_led(3,245,len(RED))              
    


def mutagen_length(path):
    try:
        audio = MP3(path)
        length = audio.info.length
        return length
    except:
        return None
def countdown(t):
    m = alsaaudio.Mixer()
    current_volume = m.getvolume()
    m.setvolume(0)
    while int(t) > 0:
        time.sleep(1)
        t = int(t) - 1
        print(t)
        #if int(t) == int(t) - 1:
        if t == 29:
            m.setvolume(10)
        #if int(t) == int(t) - 2:
        if t == 28:
            m.setvolume(20)
        #if int(t) == int(t) -3:
        if t == 27:
            m.setvolume(30)
        #if int(t) == int(t) -4:
        if t == 26:
            m.setvolume(40)
        #if int(t) == int(t) -5:
        if t == 25:
            m.setvolume(50)
        if t == 5:
            m.setvolume(40)
        if t == 4:
            m.setvolume(30)
        if t == 3:
            m.setvolume(20)
        if t == 2:
            m.setvolume(10)
        if t == 1:
            m.setvolume(5)
    print('Finish sound control')    

def main():
    ##### Config phase #####
    cap = cv2.VideoCapture(0)
    if cap.set(cv2.CAP_PROP_BUFFERSIZE, 1):  # set buffer size 
      print("Set up successful")
    else:
      print("Backend not support")
    detector = FaceDetector()
    cout = 0
    previous_emotion = 'Neutral'
    loop_script_caculate_rgb(up_step,down_step)
    SYSTEM_STATE = 0 #idle mode
    currentEmotion = 'N/A'
    IsLightAndSoundPlay = False
    IsLightScriptFinish = True
    IsSoundScriptFinish = True
    light_control = None
    play_music = None
    # create an instance of an event
    event = multiprocessing.Event()
    musicPath = ''
    m = alsaaudio.Mixer()
    current_volume = m.getvolume()
    m.setvolume(0)
    ###### Loop phase ########
    while True:
        ###### Detect emotion phase ######
        cap = cv2.VideoCapture(0)
        if cap.set(cv2.CAP_PROP_BUFFERSIZE, 1):  # set buffer size 
          print("CAMERA - Set up successful")
        else:
          print("Backend not support")
        print("-------------------" + str(cout)+ '----------------------')
        print('Read frame from camera')
        turn_off_led()
        frame = 0
        ret = 0 
        previous_emotion = currentEmotion
        currentEmotion = 'N/A'
        max_emotion = 999
        ret, frame = cap.read()
        image = frame
        cap.release()
        cv2.destroyAllWindows()

        if ret:
            print('Detecting emotion')          
            img, bboxs = detector.findFaces(image)
            max_emotion = max_result
            print("MAX" + str(max_emotion))
            ############### Determine state phase  ######################
            if bboxs:
              if (int(max_emotion) < 0) or (int(max_emotion) > 6):
                print('N/A --- Idle mode')
                SYSTEM_STATE = 0
                pass
              else:
                SYSTEM_STATE = 1
                print('INFO: Working mode')
            else:
                print("INFO: Fail at detect emotion phase")
                continue

            if SYSTEM_STATE == 0:
                print("Idle mode")

                #idle mode
            else:
                #Working mode
                #Working mode - Phase 1: prepare || Create thread to control volumn - Logging emotion
                #print('INFO: Working mode')
                #t1_volumn_control = threading.Thread(target=countdown, args=(30,))
                print("INFO: Previous emotion: " + str(previous_emotion))
                currentEmotion =  str(idx_to_class[max_emotion])
                print("INFO: Current Emotion: " + str(currentEmotion))
                
                cv2.imwrite('./test_log/' + str(currentEmotion) + '_' + str(cout)+'.jpg', img)
                #Working mode - Phase 2: Control Light and sound
                
                print("INFO: IsLightAndSoundPlay: " +str(IsLightAndSoundPlay))
                if previous_emotion != currentEmotion or (IsLightScriptFinish and IsSoundScriptFinish):
                    # Terminate process
                    # if IsLightAndSoundPlay:
                    try:
                            m.setvolume(10)
                            time.sleep(0.5)
                            m.setvolume(20)
                            time.sleep(0.5)
                            m.setvolume(30)
                            time.sleep(0.5)
                            m.setvolume(40)
                            time.sleep(0.5)
                            m.setvolume(50)
                            play_music.terminate()
                            event.clear()
                            time.sleep(1)
                            #light_control.terminate()
                            IsLightAndSoundPlay = False
                            print("INFO: ternimate music and light")
                    except:
                        print("Light and sound are not running")
                    #Setup sound path and light state
                    print("INFO: Change state of light and sound")
                    if (currentEmotion == 'Anger') or (currentEmotion == 'Disgust') or (currentEmotion == 'Fear'):
                        IsLightAndSoundPlay = True
                        musicPath = str(PATH1_HAPPY) + str(random.choice(L1_HAPPY))
                        mp3_length = mutagen_length(musicPath)
                        LightState = 'Activate_unpleasant'
                        #play_music = subprocess.Popen(["mpg123", str(musicPath)])
                        event.set()
                        #light_control = multiprocessing.Process(target=light_process, args=(event, LightState,))
                        
                        print('Mp3 duration: ' +str(mp3_length)+'(s)' )
                        print('Activate_unpleasant')

                    elif currentEmotion == 'Sadness':
                        #Unactivated unpleasant 
                        musicPath = str(PATH2_NEUTRAL) + str(random.choice(L2_NEUTRAL))
                        mp3_length = mutagen_length(musicPath)
                        #play_music = subprocess.Popen(["mpg123", str(musicPath)])

                        IsLightAndSoundPlay = True
                        LightState == 'Unactivate_unpleasant'
                        event.set()
                        #light_control = multiprocessing.Process(target=light_process, args=(event, LightState,))
                        print('Mp3 duration: ' +str(mp3_length)+'(s)' )          
                        print('Unactivate_unpleasant')              

                    elif currentEmotion == 'Happiness' or currentEmotion == 'Surprise':
                        #Activated pleasant affect
                        musicPath = str(PATH2_NEUTRAL) + str(random.choice(L2_NEUTRAL))
                        mp3_length = mutagen_length(musicPath)
                        #play_music = subprocess.Popen(["mpg123", str(musicPath)])  
                        IsLightAndSoundPlay = True
                        event.set() 
                        LightState = 'Activated_pleasant'
                        #light_control = multiprocessing.Process(target=light_process, args=(event, LightState,))
                                               

                        print('Mp3 duration: ' +str(mp3_length)+'(s)' )
                        print('activated pleasant')

                    else:
                        musicPath = str(PATH2_NEUTRAL) + str(random.choice(L2_NEUTRAL))
                        mp3_length = mutagen_length(musicPath)
                        #play_music = subprocess.Popen(["mpg123", str(musicPath)])

                        IsLightAndSoundPlay = True
                        event.set()  
                        LightState = 'Neutral'
                        #light_control = multiprocessing.Process(target=light_process, args=(event, LightState,))
                                                              
                        print('Mp3 duration: ' +str(mp3_length)+'(s)' )
                        print('Neutral')                    

                        #play_music.terminate()
                    # Start light and sound
                    
                    #light_control = multiprocessing.Process(target=light_process, args=(event, LightState,))
                    #light_control.start()
                    m.setvolume(10)
                    time.sleep(0.5)
                    m.setvolume(20)
                    time.sleep(0.5)
                    m.setvolume(30)
                    time.sleep(0.5)
                    m.setvolume(40)
                    time.sleep(0.5)
                    m.setvolume(50)                                        
                    play_music = subprocess.Popen(["mpg123", str(musicPath)])
                    
                    IsLightScriptFinish = False
                    IsSoundScriptFinish = False
                else:
                    print("Continute light and sound")
                    #continue
                if play_music is not None:
                    print("INFO: Script finish correctly")
                    IsLightScriptFinish = True
                    IsSoundScriptFinish = True
                    IsLightAndSoundPlay = False
        else:
            print('Error_camera')
            cout = cout + 1
            pass
        cout = cout + 1  
        time.sleep(16)

  
if __name__ == "__main__":
    main()
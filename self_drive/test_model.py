# test_model.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check
from basic_setup import WIDTH, HEIGHT, REGION, pre_process_image
import random

MODEL_NAME = 'autocar-0.001-alexnet-10-epochs.model'
t_time = 0.09
LR = 1e-3

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)
    
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

if __name__ == '__main__':
    print("Get ready: ")
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            screen = grab_screen(region=REGION)
            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = pre_process_image(screen)

            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
            print(prediction)

            turn_thresh = .70
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                straight()
                #print("straight")
            elif prediction[0] > turn_thresh:
                left()
                #print("left")
            elif prediction[2] > turn_thresh:
                right()
                #print("right")
            else:
                straight()
                #print("straight")

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

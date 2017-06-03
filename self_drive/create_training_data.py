# create_training_data.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
import threading
from basic_setup import WIDTH, HEIGHT, REGION, pre_process_image
import xbox_input

x_value = float(0)

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []
    
def captureX():
    """
    Grab 1st available gamepad, logging changes to the screen.
    L & R analogue triggers set the vibration motor speed.
    """
    joysticks = xbox_input.XInputJoystick.enumerate_devices()
    device_numbers = list(map(xbox_input.attrgetter('device_number'), joysticks))

    print('found %d devices: %s' % (len(joysticks), device_numbers))

    if not joysticks:
        sys.exit(0)

    j = joysticks[0]
    print('using %d' % j.device_number)
    
    @j.event
    def on_axis(axis, value):
        global x_value 
        if axis == "l_thumb_x":
            x_value = value
            #print('X', type(value), x_value)

    while True:
        j.dispatch_events()
        time.sleep(.01)


if __name__ == '__main__':

    ''' This section starts a thread and capture xbox controller input, use only when xbox controller is connected '''
    global x_value
    t = threading.Thread(target=captureX)
    t.daemon = True
    t.start()
    ''' xbox end '''
    
    print("Get ready: ")
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):

        if not paused:
            
            screen = grab_screen(region=REGION)
            screen = pre_process_image(screen)
#             print(screen)
            # resize to something a bit more acceptable for a CNN
            
            ''' Capture keyboard and use one-hot vector as output (classification) '''
            #keys = key_check()
            #output = keys_to_output(keys)
            
            ''' Capture xbox controller and use short variable as output (regression) '''
            output = x_value
            
            print(output)
            training_data.append([screen,output])
            
            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name,training_data)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


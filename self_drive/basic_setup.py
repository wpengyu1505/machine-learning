from grabscreen import grab_screen
import numpy as np
import cv2

WIDTH = 160
HEIGHT = 120

# 800x600 windowed mode
REGION = (0,20,800,620)

def pre_process_image(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_img = cv2.resize(new_img, (WIDTH,HEIGHT))
#     mask = np.zeros_like(new_img)
#     mask[int(WIDTH * 0.2):,:] = 255
#     new_img = cv2.bitwise_and(new_img, mask)
    return new_img

def test_process(img):
    new_img = img
    new_img[:,:,2] = 0
    new_img[:,:,0] = 0
    new_img[:,:,3] = 0
    # shape (601, 801, 4)
    print(np.shape(new_img))
    return new_img

if __name__ == '__main__':
    while(True):
        screen = grab_screen(region=REGION)
        screen = pre_process_image(screen)
#         screen = test_process(screen)
        cv2.imshow('image1', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
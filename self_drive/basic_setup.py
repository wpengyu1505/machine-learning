from grabscreen import grab_screen
import cv2

WIDTH = 160
HEIGHT = 120

# 800x600 windowed mode
REGION = (0,20,800,620)

if __name__ == '__main__':
    screen = grab_screen(region=REGION)
    cv2.imshow('image', screen)
    cv2.waitKey(1000)
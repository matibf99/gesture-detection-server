import urllib

import cv2
import numpy as np

from utils.gestures_detection import process_image


class ImageProcessor:
    def __init__(self, img_source):
        self.last_frame = None
        self.kill = False

        if img_source.startswith('camera:///'):
            self.cap = cv2.VideoCapture(int(img_source.replace('camera:///', '')))
        elif img_source.startswith('url:///'):
            self.url = img_source.replace('url:///', '')
        else:
            print('Incorrect source format')

    def run(self):
        print('process images...')

        if self.cap is not None:
            self.__run_camera()
        elif self.url is not None:
            self.__run_url()

    def __run_camera(self):
        print('from camera')

        while self.cap.isOpened() and self.kill is not True:
            # Read the current frame
            success, image = self.cap.read()

            # Get the current frame from the ESP32-CAM
            # img_resp = urllib.request.urlopen(url)

            # Read the current frame using numpy and opencv
            # imgnp = np.array(bytearray(img_resp.read()),dtype=np.uint8)
            # image = cv2.imdecode(imgnp,-1)

            self.last_frame = process_image(image)

    def __run_url(self):
        while self.kill is not True:
            # Get the current frame from the ESP32-CAM
            img_resp = urllib.request.urlopen(self.url)

            print('url')

            # Read the current frame using numpy and opencv
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            image = cv2.imdecode(imgnp, -1)

            self.last_frame = process_image(image)

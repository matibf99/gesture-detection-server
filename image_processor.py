import cv2

from utils.gestures_detection import process_image


class ImageProcessor:
    def __init__(self):
        self.last_frame = None
        self.cap = cv2.VideoCapture(0)

    def run(self):
        print('process images...')

        while self.cap.isOpened():
            # Read the current frame
            success, image = self.cap.read()

            # Get the current frame from the ESP32-CAM
            # img_resp = urllib.request.urlopen(url)

            # Read the current frame using numpy and opencv
            # imgnp = np.array(bytearray(img_resp.read()),dtype=np.uint8)
            # image = cv2.imdecode(imgnp,-1)

            self.last_frame = process_image(image)

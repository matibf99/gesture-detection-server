import queue
import time
import urllib
from typing import Any

import cv2
import numpy as np
import mediapipe as mp

from entities.mqtt_message import MqttMessage
from utils.cv2_utils import fill_poly_trans

from utils.colors import Colors
from utils.landmark_points import LandmarkPoints
from utils.math_utils import euclidean_distance

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                  refine_landmarks=True)


def blink_ratio(landmarks, is_right_eye):
    if is_right_eye:
        h_left = landmarks[LandmarkPoints.RIGHT_EYE_HORIZONTAL_LEFT]
        h_right = landmarks[LandmarkPoints.RIGHT_EYE_HORIZONTAL_RIGHT]
        v_top = landmarks[LandmarkPoints.RIGHT_EYE_VERTICAL_TOP]
        v_bottom = landmarks[LandmarkPoints.RIGHT_EYE_VERTICAL_BOTTOM]
    else:
        h_left = landmarks[LandmarkPoints.LEFT_EYE_HORIZONTAL_LEFT]
        h_right = landmarks[LandmarkPoints.LEFT_EYE_HORIZONTAL_RIGHT]
        v_top = landmarks[LandmarkPoints.LEFT_EYE_VERTICAL_TOP]
        v_bottom = landmarks[LandmarkPoints.LEFT_EYE_VERTICAL_BOTTOM]

        # Finding distance between horizontal and vertical points
    h_distance = euclidean_distance(h_left, h_right)
    v_distance = euclidean_distance(v_top, v_bottom)

    # Finding ratio - horizontal_distance / vertical_distance
    if v_distance > 0:
        ratio = h_distance / v_distance
    else:
        ratio = -1

    return ratio


def mouth_ratio(landmarks):
    # Calculate mouth height and width
    mouth_height = euclidean_distance(landmarks[LandmarkPoints.UPPER_LIPS_TOP],
                                      landmarks[LandmarkPoints.LOWER_LIPS_BOTTOM])
    mouth_width = euclidean_distance(landmarks[LandmarkPoints.LIP_LEFT], landmarks[LandmarkPoints.LIP_RIGHT])

    if mouth_height > 0:
        ratio = mouth_width / mouth_height
    else:
        ratio = -1

    return ratio


def raised_eyebrows(landmarks):
    left_eyebrow_to_eye_distance = euclidean_distance(landmarks[LandmarkPoints.LEFT_EYE_VERTICAL_TOP],
                                                      landmarks[LandmarkPoints.LEFT_EYEBROW_LOWER_MIDPOINT])

    right_eyebrow_to_eye_distance = euclidean_distance(landmarks[LandmarkPoints.RIGHT_EYE_VERTICAL_TOP],
                                                       landmarks[LandmarkPoints.RIGHT_EYEBROW_LOWER_MIDPOINT])

    #print(str(left_eyebrow_to_eye_distance) + " and " + str(right_eyebrow_to_eye_distance))
    threshold = 25

    return left_eyebrow_to_eye_distance > threshold or right_eyebrow_to_eye_distance > threshold


def landmark_detection(img, results, draw=False):
    img_height, img_width = img.shape[:2]

    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]

    if draw:
        [cv2.circle(img, p, 2, Colors.GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmark
    return mesh_coord


class ImageProcessor:
    img_source: str
    queue_gestures: queue
    gestures_dict: dict

    def __init__(self, img_source, queue_gestures):
        self.last_frame = None
        self.queue_gestures = queue_gestures
        self.gestures_dict = {
            'left_eye': {
                'last_closed': 0,
                'closed': False,
                'sent': False,
            },
            'right_eye': {
                'last_closed': 0,
                'closed': False,
                'sent': False,
            },
            'eyebrows': {
                'last_raised': 0,
                'raised': False,
                'sent': False,
            },
            'mouth': {
                'last_opened': 0,
                'open': False,
                'sent': False,
            }
        }

        if img_source.startswith('camera:///'):
            self.cap = cv2.VideoCapture(int(img_source.replace('camera:///', '')))
        elif img_source.startswith('url:///'):
            self.url = img_source.replace('url:///', '')
        else:
            print('Incorrect source format')

    def run(self):
        while True:
            frame = self.__get_frame()

            if frame is None:
                continue

            processed_frame = self.__process_image(frame)
            self.last_frame = processed_frame

    def __get_frame(self) -> Any:
        if self.cap is not None and self.cap.isOpened():
            return self.__get_frame_camera()
        elif self.url is not None:
            return self.__get_frame_url()
        else:
            return None

    def __get_frame_camera(self) -> Any:
        success, image = self.cap.read()
        return image

    def __get_frame_url(self) -> Any:
        # Get the current frame from the ESP32-CAM
        img_resp = urllib.request.urlopen(self.url)

        print('url')

        # Read the current frame using numpy and opencv
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        image = cv2.imdecode(imgnp, -1)

        return image

    def __process_image(self, image):
        image_height, frame_width = image.shape[:2]

        # Get the start time to calculate FPS at the end
        start_time = time.time()

        # Flip the image horizontally for a selfie-view display
        # Also convert the image from BRG to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Process the image using face_mesh
        results = face_mesh.process(image)

        # Convert back the image from RGB to BRG so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if results.multi_face_landmarks:
            mesh_coords = landmark_detection(image, results, False)
            # image = fillPolyTrans(image, [mesh_coords[p] for p in LandmarkPoints.FACE_OVAL], Colors.WHITE, opacity=0.4)
            image = fill_poly_trans(image, [mesh_coords[p] for p in LandmarkPoints.LEFT_EYE], Colors.GREEN,
                                    opacity=0.4)
            image = fill_poly_trans(image, [mesh_coords[p] for p in LandmarkPoints.RIGHT_EYE], Colors.RED,
                                    opacity=0.4)
            image = fill_poly_trans(image, [mesh_coords[p] for p in LandmarkPoints.LEFT_EYEBROW], Colors.ORANGE,
                                    opacity=0.4)
            image = fill_poly_trans(image, [mesh_coords[p] for p in LandmarkPoints.RIGHT_EYEBROW], Colors.ORANGE,
                                    opacity=0.4)
            image = fill_poly_trans(image, [mesh_coords[p] for p in LandmarkPoints.LIPS], Colors.BLACK, opacity=0.3)

            left_eye_ratio = blink_ratio(mesh_coords, is_right_eye=False)
            right_eye_ratio = blink_ratio(mesh_coords, is_right_eye=True)

            left_eye_closed = left_eye_ratio > 5.5
            right_eye_closed = right_eye_ratio > 5.5

            cv2.putText(image, f'L-Blink: {round(left_eye_ratio, 2)}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        Colors.GREEN, 2)
            cv2.putText(image, f'R-Blink: {round(right_eye_ratio, 2)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        Colors.RED, 2)

            cv2.putText(image, f'L-Closed' if left_eye_closed else f'L-Open', (20, 190), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, Colors.GREEN, 2)
            cv2.putText(image, f'R-Closed' if right_eye_closed else f'R-Open', (20, 230), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, Colors.RED, 2)
            cv2.putText(image, f'Eyes closed' if (right_eye_closed and left_eye_closed) else f'Eyes open',
                        (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, Colors.YELLOW, 2)

            m_ratio = mouth_ratio(mesh_coords)

            mouth_open = m_ratio < 2
            cv2.putText(image, f'Mouth: {round(m_ratio, 2)}', (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        Colors.PINK, 2)
            cv2.putText(image, f'M-Open' if mouth_open else f'M-Closed', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        Colors.PINK, 2)

            eyebrows_raised = raised_eyebrows(mesh_coords)

            cv2.putText(image, f'EB-Raised' if eyebrows_raised else f'EB-Not raised', (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        Colors.PURPLE, 2)

            # colorBackgroundText(image,  f'L-Blink: {round(left_eye_ratio,2)}', cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 100), 2, Colors.YELLOW, pad_x=6, pad_y=6)
            # colorBackgroundText(image,  f'R-Blink: {round(right_eye_ratio,2)}', cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 130), 2, Colors.YELLOW, pad_x=6, pad_y=6)

            self.__send_gestures(left_eye_closed=left_eye_closed,
                                 right_eye_closed=right_eye_closed,
                                 mouth_open=mouth_open,
                                 eyebrows_raised=eyebrows_raised)

        # Get the end time and calculate the total time
        end_time = time.time()
        total_time = end_time - start_time

        fps = 1 / total_time

        # Print the FPS on to the image
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        return image

    def __send_gestures(self, left_eye_closed, right_eye_closed, mouth_open, eyebrows_raised):
        # Detect mouth gesture
        if mouth_open != self.gestures_dict['mouth']['open']:
            self.gestures_dict['mouth']['open'] = mouth_open
            self.gestures_dict['mouth']['last_opened'] = time.time()

            if self.gestures_dict['mouth']['sent']:
                message = MqttMessage(topic='gesture/mouth', payload="0")
                self.queue_gestures.put_nowait(message)

            self.gestures_dict['mouth']['sent'] = False

        # Detect left eye gesture
        if left_eye_closed != self.gestures_dict['left_eye']['closed']:
            self.gestures_dict['left_eye']['closed'] = left_eye_closed
            self.gestures_dict['left_eye']['last_closed'] = time.time()

            if self.gestures_dict['left_eye']['sent']:
                message = MqttMessage(topic='gesture/left_eye', payload="0")
                self.queue_gestures.put_nowait(message)

            self.gestures_dict['left_eye']['sent'] = False

        # Detect right eye gesture
        if right_eye_closed != self.gestures_dict['right_eye']['closed']:
            self.gestures_dict['right_eye']['closed'] = right_eye_closed
            self.gestures_dict['right_eye']['last_closed'] = time.time()

            if self.gestures_dict['right_eye']['sent']:
                message = MqttMessage(topic='gesture/right_eye', payload="0")
                self.queue_gestures.put_nowait(message)

            self.gestures_dict['right_eye']['sent'] = False

        # Detect eyebrows raised gesture
        if eyebrows_raised != self.gestures_dict['eyebrows']['raised']:
            self.gestures_dict['eyebrows']['raised'] = eyebrows_raised
            self.gestures_dict['eyebrows']['last_raised'] = time.time()

            if self.gestures_dict['eyebrows']['sent']:
                message = MqttMessage(topic='gesture/eyebrows', payload="0")
                self.queue_gestures.put_nowait(message)

            self.gestures_dict['eyebrows']['sent'] = False

        # Calculate time since gesture started
        actual_time = time.time()
        elapsed_time_left_eye = actual_time - self.gestures_dict['left_eye']['last_closed']
        elapsed_time_right_eye = actual_time - self.gestures_dict['right_eye']['last_closed']
        elapsed_time_mouth = actual_time - self.gestures_dict['mouth']['last_opened']
        elapsed_time_eyebrows_raised = actual_time - self.gestures_dict['eyebrows']['last_raised']

        # Detect if enough time has passed for mouth gesture
        if elapsed_time_mouth >= 1 and self.gestures_dict['mouth']['open'] and not self.gestures_dict['mouth']['sent']:
            self.gestures_dict['mouth']['sent'] = True
            message = MqttMessage(topic='gesture/mouth', payload="1")
            self.queue_gestures.put_nowait(message)

        # Detect if enough time has passed for left eye gesture
        if elapsed_time_left_eye >= 1 and self.gestures_dict['left_eye']['closed'] \
                and not self.gestures_dict['left_eye']['sent']:
            self.gestures_dict['left_eye']['sent'] = True
            message = MqttMessage(topic='gesture/left_eye', payload="1")
            self.queue_gestures.put_nowait(message)

        # Detect if enough time has passed for right eye gesture
        # print(self.gestures_dict['right_eye'])
        if elapsed_time_right_eye >= 1 and self.gestures_dict['right_eye']['closed'] \
                and not self.gestures_dict['right_eye']['sent']:
            self.gestures_dict['right_eye']['sent'] = True
            message = MqttMessage(topic='gesture/right_eye', payload="1")
            self.queue_gestures.put_nowait(message)

        # Detect if enough time has passed for eyebrows gesture
        if elapsed_time_eyebrows_raised >= 1 and self.gestures_dict['eyebrows']['raised'] \
                and not self.gestures_dict['eyebrows']['sent']:
            self.gestures_dict['eyebrows']['sent'] = True
            message = MqttMessage(topic='gesture/eyebrows', payload="1")
            self.queue_gestures.put_nowait(message)

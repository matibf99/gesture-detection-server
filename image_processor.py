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


def landmark_detection(img, results, draw=False):
    img_height, img_width = img.shape[:2]

    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]

    if draw:
        [cv2.circle(img, p, 2, Colors.GREEN, -1) for p in mesh_coord]

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
                'last': 0,
                'previous_detected': False,
                'detected': False,
                'ratio': 0,
                'sent': False,
            },
            'right_eye': {
                'last': 0,
                'previous_detected': False,
                'detected': False,
                'ratio': 0,
                'sent': False,
            },
            'eyebrows_raised': {
                'last': 0,
                'previous_detected': False,
                'detected': False,
                'ratio': 0,
                'sent': False,
            },
            'mouth_open': {
                'last': 0,
                'previous_detected': False,
                'detected': False,
                'ratio': 0,
                'sent': False,
            },
            'mouth_right': {
                'last': 0,
                'previous_detected': False,
                'detected': False,
                'ratio': 0,
                'sent': False,
            },
            'mouth_left': {
                'last': 0,
                'previous_detected': False,
                'detected': False,
                'ratio': 0,
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
            mesh_cords = landmark_detection(image, results)
            image = self.__draw_landmarks(image, mesh_cords)

            self.__blink_left_eye(mesh_cords)
            self.__blink_right_eye(mesh_cords)
            self.__raised_eyebrows(mesh_cords)
            self.__mouth_open(mesh_cords)
            self.__mouth_left(mesh_cords)
            self.__mouth_right(mesh_cords)

            self.__detect_and_print_text_gesture(image=image,
                                                 gesture_name='LE-Blink',
                                                 gesture_dict=self.gestures_dict['left_eye'],
                                                 topic="gesture/left_eye",
                                                 x_pos=20, y_pos=110,
                                                 text_color=Colors.GREEN)

            self.__detect_and_print_text_gesture(image=image,
                                                 gesture_name='RE-Blink',
                                                 gesture_dict=self.gestures_dict['right_eye'],
                                                 topic="gesture/right_eye",
                                                 x_pos=20, y_pos=150,
                                                 text_color=Colors.GREEN)

            self.__detect_and_print_text_gesture(image=image,
                                                 gesture_name='EB-Raised',
                                                 gesture_dict=self.gestures_dict['eyebrows_raised'],
                                                 topic="gesture/eyebrows_raised",
                                                 x_pos=20, y_pos=190,
                                                 text_color=Colors.PURPLE)

            self.__detect_and_print_text_gesture(image=image,
                                                 gesture_name='M-Open',
                                                 gesture_dict=self.gestures_dict['mouth_open'],
                                                 topic="gesture/mouth_open",
                                                 x_pos=20, y_pos=230,
                                                 text_color=Colors.RED)

            self.__detect_and_print_text_gesture(image=image,
                                                 gesture_name='M-Left',
                                                 gesture_dict=self.gestures_dict['mouth_left'],
                                                 topic="gesture/mouth_left",
                                                 x_pos=20, y_pos=270,
                                                 text_color=Colors.YELLOW)

            self.__detect_and_print_text_gesture(image=image,
                                                 gesture_name='M-Right',
                                                 gesture_dict=self.gestures_dict['mouth_right'],
                                                 topic="gesture/mouth_right",
                                                 x_pos=20, y_pos=310,
                                                 text_color=Colors.YELLOW)

        # Get the end time and calculate the total time
        end_time = time.time()
        total_time = end_time - start_time

        fps = 1 / total_time

        # Print the FPS on to the image
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        return image

    def __draw_landmarks(self, image, mesh_cords):
        # image = fillPolyTrans(image, [mesh_cords[p] for p in LandmarkPoints.FACE_OVAL], Colors.WHITE, opacity=0.4)
        image = fill_poly_trans(image, [mesh_cords[p] for p in LandmarkPoints.LEFT_EYE], Colors.GREEN, opacity=0.4)
        image = fill_poly_trans(image, [mesh_cords[p] for p in LandmarkPoints.RIGHT_EYE], Colors.RED, opacity=0.4)
        image = fill_poly_trans(image, [mesh_cords[p] for p in LandmarkPoints.LEFT_EYEBROW], Colors.ORANGE, opacity=0.4)
        image = fill_poly_trans(image, [mesh_cords[p] for p in LandmarkPoints.RIGHT_EYEBROW], Colors.ORANGE, opacity=0.4)
        image = fill_poly_trans(image, [mesh_cords[p] for p in LandmarkPoints.LIPS], Colors.BLACK, opacity=0.3)
        cv2.circle(image, mesh_cords[LandmarkPoints.NOSE_POINT], 2, Colors.WHITE, -1)
        cv2.circle(image, mesh_cords[LandmarkPoints.NOSE_UNDER_POINT], 2, Colors.RED, -1)
        cv2.circle(image, mesh_cords[LandmarkPoints.LIP_LEFT], 2, Colors.WHITE, -1)
        cv2.circle(image, mesh_cords[LandmarkPoints.LIP_RIGHT], 2, Colors.WHITE, -1)

        return image

    def __detect_and_print_text_gesture(self, image, gesture_name, gesture_dict, topic, x_pos, y_pos,
                                        text_color=Colors.BLACK):
        cv2.putText(image, f'{gesture_name} - Y - {round(gesture_dict["ratio"], 2)}' if gesture_dict["detected"]
                    else f'{gesture_name} - N - {round(gesture_dict["ratio"], 2)}', (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        if gesture_dict['detected'] != gesture_dict['previous_detected']:
            gesture_dict['previous_detected'] = gesture_dict['detected']
            gesture_dict['last'] = time.time()

            if gesture_dict['sent']:
                message = MqttMessage(topic=topic, payload="0")
                self.queue_gestures.put_nowait(message)

            gesture_dict['sent'] = False

        # Calculate time since gesture started
        actual_time = time.time()
        elapsed_time = actual_time - gesture_dict['last']

        # if 1 < elapsed_time < 1000 and gesture_dict['detected']:
        #    print(gesture_name + " -1 " + str(elapsed_time))

        # Detect if enough time has passed for mouth gesture
        if elapsed_time >= 1 and gesture_dict['detected'] and not gesture_dict['sent']:
            gesture_dict['sent'] = True
            message = MqttMessage(topic=topic, payload="1")
            self.queue_gestures.put_nowait(message)

    def __blink(self, landmarks, is_right_eye):
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

        self.gestures_dict['right_eye' if is_right_eye else 'left_eye']['detected'] = ratio > 8
        self.gestures_dict['right_eye' if is_right_eye else 'left_eye']['ratio'] = ratio

    def __blink_left_eye(self, landmarks):
        self.__blink(landmarks, is_right_eye=False)

    def __blink_right_eye(self, landmarks):
        self.__blink(landmarks, is_right_eye=True)

    def __mouth_open(self, landmarks):
        mouth_height = euclidean_distance(landmarks[LandmarkPoints.UPPER_LIPS_TOP],
                                          landmarks[LandmarkPoints.LOWER_LIPS_BOTTOM])
        mouth_width = euclidean_distance(landmarks[LandmarkPoints.LIP_LEFT], landmarks[LandmarkPoints.LIP_RIGHT])

        if mouth_height > 0:
            ratio = mouth_width / mouth_height
        else:
            ratio = -1

        self.gestures_dict['mouth_open']['detected'] = ratio < 2
        self.gestures_dict['mouth_open']['ratio'] = ratio

    def __raised_eyebrows(self, landmarks):
        left_eyebrow_to_eye_distance = euclidean_distance(landmarks[LandmarkPoints.LEFT_EYE_VERTICAL_TOP],
                                                          landmarks[LandmarkPoints.LEFT_EYEBROW_UPPER_MIDPOINT])
        right_eyebrow_to_eye_distance = euclidean_distance(landmarks[LandmarkPoints.RIGHT_EYE_VERTICAL_TOP],
                                                           landmarks[LandmarkPoints.RIGHT_EYEBROW_UPPER_MIDPOINT])

        left_eyebrow_to_face_top_distance = euclidean_distance(landmarks[LandmarkPoints.FACE_TOP_LEFT_EYEBROW],
                                                      landmarks[LandmarkPoints.LEFT_EYEBROW_UPPER_MIDPOINT])
        right_eyebrow_to_face_top_distance = euclidean_distance(landmarks[LandmarkPoints.FACE_TOP_RIGHT_EYEBROW],
                                                       landmarks[LandmarkPoints.RIGHT_EYEBROW_UPPER_MIDPOINT])

        ratio_left = left_eyebrow_to_face_top_distance / left_eyebrow_to_eye_distance
        ratio_right = right_eyebrow_to_face_top_distance / right_eyebrow_to_eye_distance

        ratio = max(ratio_left, ratio_right)

        threshold = 1

        self.gestures_dict['eyebrows_raised']['detected'] = ratio < threshold
        self.gestures_dict['eyebrows_raised']['ratio'] = ratio

    def __mouth_side(self, landmarks, is_right_side=True):
        mouth_top_to_nose = euclidean_distance(landmarks[LandmarkPoints.UPPER_LIPS_TOP],
                                               landmarks[LandmarkPoints.NOSE_UNDER_POINT])

        mouth_side_to_middle = euclidean_distance(
            landmarks[LandmarkPoints.UPPER_LIPS_TOP],
            landmarks[LandmarkPoints.LIP_RIGHT if is_right_side else LandmarkPoints.LIP_LEFT])

        ratio = mouth_top_to_nose / mouth_side_to_middle
        threshold = 0.55

        self.gestures_dict['mouth_right' if is_right_side else 'mouth_left']['detected'] = \
            ratio < threshold and not self.gestures_dict['mouth_open']['detected']
        self.gestures_dict['mouth_right' if is_right_side else 'mouth_left']['ratio'] = ratio

    def __mouth_left(self, landmarks):
        self.__mouth_side(landmarks, is_right_side=False)

    def __mouth_right(self, landmarks):
        self.__mouth_side(landmarks, is_right_side=True)

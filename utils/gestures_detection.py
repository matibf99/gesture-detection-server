import time
import cv2
import mediapipe as mp

from utils.colors import Colors
from utils.cv2_utils import fill_poly_trans
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


def process_image(image):
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

        mouth_closed = m_ratio > 2
        cv2.putText(image, f'Mouth: {round(m_ratio, 2)}', (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    Colors.PINK, 2)
        cv2.putText(image, f'M-Closed' if mouth_closed else f'M-Open', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    Colors.PINK, 2)

        # colorBackgroundText(image,  f'L-Blink: {round(left_eye_ratio,2)}', cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 100), 2, Colors.YELLOW, pad_x=6, pad_y=6)
        # colorBackgroundText(image,  f'R-Blink: {round(right_eye_ratio,2)}', cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 130), 2, Colors.YELLOW, pad_x=6, pad_y=6)

    # Get the end time and calculate the total time
    end_time = time.time()
    total_time = end_time - start_time

    fps = 1 / total_time

    # Print the FPS on to the image
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    return image


def landmark_detection(img, results, draw=False):
    img_height, img_width = img.shape[:2]

    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]

    if draw:
        [cv2.circle(img, p, 2, Colors.GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmark
    return mesh_coord

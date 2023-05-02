# Landmark points: https://github.com/ManuelTS/augmentedFaceMeshIndices
class LandmarkPoints:
    # Face boundaries
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # Lips
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40,
            39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
    LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
    UPPER_LIPS_TOP = UPPER_LIPS[4]
    UPPER_LIPS_BOTTOM = UPPER_LIPS[13]
    LOWER_LIPS_TOP = LOWER_LIPS[16]
    LOWER_LIPS_BOTTOM = LOWER_LIPS[5]
    LIP_LEFT = LIPS[0]
    LIP_RIGHT = LIPS[10]

    # Left eye
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    LEFT_EYE_HORIZONTAL_LEFT = LEFT_EYE[8]
    LEFT_EYE_HORIZONTAL_RIGHT = LEFT_EYE[0]
    LEFT_EYE_VERTICAL_TOP = LEFT_EYE[12]
    LEFT_EYE_VERTICAL_BOTTOM = LEFT_EYE[4]

    # Right eye
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_EYE_HORIZONTAL_LEFT = RIGHT_EYE[8]
    RIGHT_EYE_HORIZONTAL_RIGHT = RIGHT_EYE[0]
    RIGHT_EYE_VERTICAL_TOP = RIGHT_EYE[12]
    RIGHT_EYE_VERTICAL_BOTTOM = RIGHT_EYE[4]
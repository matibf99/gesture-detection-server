import cv2
import numpy as np


def fill_poly_trans(img, points, color, opacity):
    """
    @param img: (mat) input image, where shape is drawn.
    @param points: list [tuples(int, int) these are the points custom shape,FillPoly
    @param color: (tuples (int, int, int)
    @param opacity:  it is transparency of image.
    @return: img(mat) image with rectangle draw.
    """
    list_to_np_array = np.array(points, dtype=np.int32)
    overlay = img.copy()  # coping the image
    cv2.fillPoly(overlay, [list_to_np_array], color)
    new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    # print(points_list)
    img = new_img
    cv2.polylines(img, [list_to_np_array], True, color, 1, cv2.LINE_AA)
    return img


def color_background_text(img, text, font, font_scale, text_pos, text_thickness=1, text_color=(0, 255, 0),
                          bg_color=(0, 0, 0), pad_x=3, pad_y=3):
    """
    Draws text with background, with  control transparency
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param font_scale: (double) the size of text, how big it should be.
    @param text_pos: tuple(x,y) position where you want to draw text
    @param text_thickness:(int) fonts weight, how bold it should be
    @param text_color: tuple(BGR), values -->0 to 255 each
    @param bg_color: tuple(BGR), values -->0 to 255 each
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels) 1 to 1.0 (), controls transparency of  text background
    @return: img(mat) with draw with background
    """
    # getting the text size
    (t_w, t_h), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
    x, y = text_pos

    # draw rectangle
    cv2.rectangle(img, (x - pad_x, y + pad_y), (x + t_w + pad_x, y - t_h - pad_y), bg_color, -1)

    # draw in text
    cv2.putText(img, text, text_pos, font, font_scale, text_color, text_thickness)

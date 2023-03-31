from cvu.detector.yolov5 import Yolov5
import cv2
import numpy as np
from skimage.metrics import structural_similarity


CLASSES_UI = [
    "BackgroundImage",
    "Bottom_Navigation",
    "Card",
    "CheckBox",
    "Checkbox",
    "CheckedTextView",
    "Drawer",
    "EditText",
    "Icon",
    "Image",
    "Map",
    "Modal",
    "Multi_Tab",
    "PageIndicator",
    "Remember",
    "Spinner",
    "Switch",
    "Text",
    "TextButton",
    "Toolbar",
    "UpperTaskBar",
]


class DetectorMenuItems:
    def __init__(self) -> None:
        self.model = Yolov5(
            classes=CLASSES_UI,
            backend="torch",
            weight="models/best25.torchscript",
            device="cpu",
            input_shape=640,
        )

    # def __call__(self, img: Image) -> list:
    #     pass


def get_changed_region(img_in: np.ndarray, img_out: np.ndarray):
    img_in_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    img_out_gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(img_in_gray, img_out_gray, full=True)

    # если выполняется условие ниже, то считаем, что клик по бургер-меню не получился
    if (1 - score) < 0.005:
        return None, score * 100

    diff = (diff * 255).astype("uint8")
    # diff_box = cv2.merge([diff, diff, diff])

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # найдем контур максимального размера
    max_c_size = 0
    max_c = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_c_size:
            max_c_size = area
            max_c = c

    if type(max_c) != type(None):
        x, y, w, h = cv2.boundingRect(max_c)
        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }, score * 100

    return {}, score * 100

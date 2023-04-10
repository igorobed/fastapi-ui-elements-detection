from cvu.detector.yolov5 import Yolov5
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity
from cvu.utils.draw import draw_bbox
from PIL import Image
from .utils import (
    convert_from_image_to_cv2,
    convert_from_cv2_to_image,
)


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


class DetectorItems:
    def __init__(self) -> None:
        self.model = Yolov5(
            classes=CLASSES_UI,
            backend="torch",
            weight="models/best25.torchscript",
            device="cpu",
            input_shape=640,
        )

    def __call__(self, img: Image, draw_lst=CLASSES_UI):
        img = convert_from_image_to_cv2(img)
        prediction = self.model(img)
        found_elements = {name: [] for name in CLASSES_UI}
        for item in prediction:
            found_elements[item.class_name].append(
                {
                "top_left": list(map(int, item.bbox[:2])),
                "bottom_right": list(map(int, item.bbox[2:])),
                }
            )
            if item.class_name in draw_lst:
                # draw_bbox(img, item.bbox, color=(0, 0, 255))
                draw_bbox(img, item.bbox, title=item.class_name)
        
        img = convert_from_cv2_to_image(img)

        return img, found_elements
import cv2
import numpy as np
from PIL import Image


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def show_img(img: np.ndarray) -> None:
    cv2.imshow("img", img)
    cv2.waitKey(0)
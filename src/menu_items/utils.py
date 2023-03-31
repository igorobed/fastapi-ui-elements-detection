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


def custom_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return img[y : y + h, x : x + w]


# с этой вещью могут быть проблемы
def crop_head(img: np.ndarray, cropped_x_y) -> np.ndarray:
    """
    Обрезаем верхушку с временем, уровнем заряда,показателями телефона
    и полем с адресом страницы
    """
    cropped_x_y["y"] += 140
    return img[140:, :]


def get_squares(rects: list[list[list[int]]]) -> list[int]:
    res = []
    for rect in rects:
        res.append((rect[1][0] - rect[0][0]) * (rect[1][1] - rect[0][1]))
    
    return res

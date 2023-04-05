from cvu.detector.yolov5 import Yolov5
from cvu.utils.draw import draw_bbox
import numpy as np
from PIL import Image
from .utils import (
    # my_logger,
    convert_from_cv2_to_image,
    convert_from_image_to_cv2,
)


class DetectorBurgerMenu:
    def __init__(self) -> None:
        self.model = Yolov5(
            classes=["menu", "close"],
            backend="torch",
            weight="models/best_menu_close.torchscript",
            device="cpu",
            input_shape=640,
        )

    def __call__(self, img: Image) -> np.ndarray:
        img = convert_from_image_to_cv2(img)
        preds = self.model(img)
        # my_logger.info("The model has processed the incoming image")

        # если алгоритм не обнаружил меню на изображении,
        # то просто возвращаем исходный объект

        found_elements = {}
        # необходимо вернуть максимальный менюшный элемент
        # необходимо вернуть все найденные крестики

        menu_items = [item for item in preds if item.class_name == "menu"]
        close_items = [item for item in preds if item.class_name == "close"]

        if len(menu_items) > 0:

            # модель может найти несколько элементов похожих на меню
            # необходимо выбрать наиболее вероятный
            max_conf = max([item.confidence for item in menu_items])
            for pred in menu_items:
                if pred.confidence == max_conf:
                    best_pred = pred
                    break

            # отобразим область с обнаруженным объектом
            # но сначала проверим, что найденный элемент не распологается снизу
            # иначе у нас произошло ложноположительное срабатывание
            if best_pred.bbox[1] < 2 * (img.shape[0] / 3):
                found_elements["burger_menu"] = {
                    "top_left": list(map(int, best_pred.bbox[:2])),
                    "bottom_right": list(map(int, best_pred.bbox[2:])),
                }
                draw_bbox(img, best_pred.bbox, color=(0, 0, 255))
        else:
            # my_logger.info("The model did not find the desired object")
            found_elements["burger_menu"] = {}
        
        found_elements["close"] = []
        if len(close_items) > 0:
            for pred in close_items:
                found_elements["close"].append(
                    {
                    "top_left": list(map(int, pred.bbox[:2])),
                    "bottom_right": list(map(int, pred.bbox[2:])),
                    }
                    )
                draw_bbox(img, pred.bbox, color=(0, 0, 255))

        img = convert_from_cv2_to_image(img)

        return img, found_elements

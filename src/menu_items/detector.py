from cvu.detector.yolov5 import Yolov5
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image
from .utils import (
    convert_from_image_to_cv2,
    convert_from_cv2_to_image,
    crop_head,
    custom_crop,
    get_squares,
)
from sklearn.cluster import AgglomerativeClustering


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

    def __call__(self, img_b: Image, img_a: Image):
        img_b, img_a = convert_from_image_to_cv2(img_b), convert_from_image_to_cv2(img_a)
        # чтобы вернуться к оригинальным координатам
        cropped_x_y_in = {"x": 0, "y": 0}

        cropped_x_y_out = {"x": 0, "y": 0}

        img_b_crop, img_a_crop = crop_head(img_b, cropped_x_y=cropped_x_y_in), \
            crop_head(img_a, cropped_x_y=cropped_x_y_out)
        
        menu_box, score = get_changed_region(img_b_crop, img_a_crop)
        # score выдает процент схожести двух изображений
        if menu_box is None:
            # print(f"Same image: {score}")
            return {
                "status": "success",
                "data": menu_box,
                "details": f"Same image: {score}",
            }
        elif menu_box == {}:
            # print("Menu field not detected")
            return {
                "status": "success",
                "data": menu_box,
                "details": "Menu field not detected",
            }
    
        # вырезаем область, притерпевшую изменения
        cropped_x_y_out["x"] += menu_box["x"]
        cropped_x_y_out["y"] += menu_box["y"]
        menu_box_img = custom_crop(img_a_crop, **menu_box)

        # пропускаем полученное изображение через детектор
        predictions_crop = self.model(menu_box_img)

        # модель для поиска кластеров текстов в обрезанном menu_box_img
        clustering = AgglomerativeClustering(
            n_clusters=2,
            affinity="manhattan",
            linkage="complete",
        )

        list_bboxes = []  # предсказанные боксы с текстом
        list_bboxes_tl = []  # датасет с x-координатами для кластеризации

        # находим боксы с текстом и формируем датасет для кластеризации
        for item in predictions_crop:
            if item.class_name == "Text":
                list_bboxes.append(list(map(int, item.bbox)))
                temp = list(map(int, item.bbox[:2]))
                temp[1] = 0  # нужна только координата x
                list_bboxes_tl.append(temp)

        # находим два кластера
        clustering.fit(list_bboxes_tl)

        # сопоставим номер кластера и области относящиеся к нему
        unions = {}
        for idx, label in enumerate(clustering.labels_):
            if label in unions:
                unions[label].append(list_bboxes[idx])
            else:
                unions[label] = [list_bboxes[idx]]

        # найдем общие координаты двух областей для каждого кластера
        union_rectangles = []
        for key, items in unions.items():
            tl, br = items[0][:2], items[0][2:]

            if len(items) == 1:
                union_rectangles.append([tl, br])
                continue

            top_left_x_y, back_right_x_y = tl, br

            min_x = top_left_x_y[0]
            max_x = back_right_x_y[0]

            for box in items[1:]:
                tl, br = box[:2], box[2:]

                if tl[1] < top_left_x_y[1]:
                    top_left_x_y = tl

                if br[1] > back_right_x_y[1]:
                    back_right_x_y = br

                if back_right_x_y[0] > max_x:
                    max_x = back_right_x_y[0]

                if top_left_x_y[0] < min_x:
                    min_x = top_left_x_y[0]

            top_left_x_y = [min_x, top_left_x_y[1]]
            back_right_x_y = [max_x, back_right_x_y[1]]
            union_rectangles.append([top_left_x_y, back_right_x_y])

        # найдем область с максимальной площадью
        # обозначим как максимальную область область меньше,
        # если разница между двумя областями не превышает 15%
        # и у текущей максимальной ширина больше высоты
        squares = np.array(get_squares(union_rectangles))
        idx_max_sq, idx_min_sq = np.argmax(squares), np.argmin(squares)

        max_rectangle = union_rectangles[idx_max_sq]

        max_sq, min_sq = squares[idx_max_sq], squares[idx_min_sq]
        max_rect, min_rect = union_rectangles[idx_max_sq], union_rectangles[idx_min_sq]

        min_height, min_width = min_rect[1][1] - min_rect[0][1], min_rect[1][0] - min_rect[0][0]
        max_height, max_width = max_rect[1][1] - max_rect[0][1], max_rect[1][0] - max_rect[0][0]

        if (1 - float(min_sq) / max_sq) < 0.15 and \
        max_width > max_height and \
        min_width <= min_height:
            max_rectangle = min_rect

        # необходимо уточнить спиок элементов принадлежащих
        # максимальной области
        # + восстановить их оригинальные координаты 
        result_text_regions = []
        for item in predictions_crop:
            if item.class_name == "Text":
                x, y = int(item.bbox[0]), int(item.bbox[1])
                if x >= (max_rectangle[0][0] - 10) and y >= (max_rectangle[0][1] - 10):
                    temp = item.bbox
                    temp[0] += cropped_x_y_out["x"]
                    temp[2] += cropped_x_y_out["x"]
                    temp[1] += cropped_x_y_out["y"]
                    temp[3] += cropped_x_y_out["y"]

                    cv2.rectangle(
                        img_a,
                        list(map(int, temp[:2])),
                        list(map(int, temp[2:])),
                        (0, 0, 255),
                        2,
                    )

                    result_text_regions.append(temp)

        # результаты нужно сортировать по оси y
        result_text_regions.sort(key=lambda x: x[1])

        result = []
        for item in result_text_regions:
            temp = {}
            temp["top_left"] = list(map(int, item[:2]))
            temp["bottom_right"] = list(map(int, item[2:]))
            result.append(temp)
        
        return img_a, result
        

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

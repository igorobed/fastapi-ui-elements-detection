from fastapi import APIRouter, File, UploadFile, Response
from .schemas import MenuList
import cv2
from .utils import (
    show_img,
    convert_from_image_to_cv2,
    convert_from_cv2_to_image,
    crop_head,
    custom_crop,
    get_squares,
)

import base64
from io import BytesIO
from PIL import Image
import numpy as np

from .detector import DetectorMenuItems, get_changed_region

from sklearn.cluster import AgglomerativeClustering

router = APIRouter(
    prefix="/detect_menu_items",
    tags=["Menu items"],
)

detector = DetectorMenuItems()


@router.post("/")
async def get_menu_items(data: MenuList):
    """
    Детекция элементов меню, после нажатия на бургер-меню
    """
    img_b, img_a = data.img_before_base64, data.img_after_base64
    img_b, img_a = base64.b64decode(img_b), base64.b64decode(img_a)
    img_b, img_a = Image.open(BytesIO(img_b)), Image.open(BytesIO(img_a))

    # img, elements = detector(img_b, img_a)
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
    predictions_crop = detector.model(menu_box_img)
    
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

    # show_img(img_a)  # для отладки

    # результаты нужно сортировать по оси y
    result_text_regions.sort(key=lambda x: x[1])

    result = []
    for item in result_text_regions:
        temp = {}
        temp["top_left"] = list(map(int, item[:2]))
        temp["bottom_right"] = list(map(int, item[2:]))
        result.append(temp)

    return {
            "status": "success",
            "data": {"menu_items": result},
            "details": None,
        }


@router.post("/test_json")
async def get_menu_items(img_before: UploadFile = File(...), img_after: UploadFile = File(...)):
    """
    Тестирование детекции элементов меню с возвращением координат элементов
    """
    img_b, img_a = Image.open(img_before.file), Image.open(img_after.file)

    # img, elements = detector(img_b, img_a)
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
    predictions_crop = detector.model(menu_box_img)
    
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


    if (1 - float(min_sq) / max_sq) < 0.25 and \
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

    return {
            "status": "success",
            "data": {"menu_items": result},
            "details": None,
        }


@router.post("/test_img")
async def get_menu_items(img_before: UploadFile = File(...), img_after: UploadFile = File(...)):
    """
    Тестирование детекции элементов меню с возвращением изображения
    """
    img_b, img_a = Image.open(img_before.file), Image.open(img_after.file)

    # img, elements = detector(img_b, img_a)
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
    predictions_crop = detector.model(menu_box_img)
    
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


    if (1 - float(min_sq) / max_sq) < 0.25 and \
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

    img_a = convert_from_cv2_to_image(img_a)

    img_byte_arr = BytesIO()
    img_a.save(img_byte_arr, format='PNG')

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

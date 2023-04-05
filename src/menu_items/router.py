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

    img, elements = detector(img_b, img_a)

    return {
            "status": "success",
            "data": {"menu_items": elements},
            "details": None,
        }


@router.post("/test_json")
async def get_menu_items(img_before: UploadFile = File(...), img_after: UploadFile = File(...)):
    """
    Тестирование детекции элементов меню с возвращением координат элементов
    """
    img_b, img_a = Image.open(img_before.file), Image.open(img_after.file)

    _, elements = detector(img_b, img_a)

    return {
            "status": "success",
            "data": {"menu_items": elements},
            "details": None,
        }


@router.post("/test_img")
async def get_menu_items(img_before: UploadFile = File(...), img_after: UploadFile = File(...)):
    """
    Тестирование детекции элементов меню с возвращением изображения
    """
    img_b, img_a = Image.open(img_before.file), Image.open(img_after.file)

    img, _ = detector(img_b, img_a)
    
    img = convert_from_cv2_to_image(img)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

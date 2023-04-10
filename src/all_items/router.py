from fastapi import APIRouter, File, UploadFile, Response
from .utils import (
    convert_from_image_to_cv2,
    show_img,
)
from .detector import DetectorItems

import base64
from io import BytesIO
from PIL import Image

router = APIRouter(
    prefix="/detect_all_items",
    tags=["Burger all items"],
)

detector = DetectorItems()


@router.post("/test_json")
async def get_burger_menu(data: UploadFile = File(...)):
    """
    Тестирование детекции всевозможных элементов
    """
    img = Image.open(data.file)
    img, elements = detector(img)

    return {
        "status": "success",
        "data": elements,
        "details": None,
    }


@router.post("/test_img")
async def get_burger_menu(data: UploadFile = File(...)):
    """
    Тестирование детекции всевозможных элементов
    """
    img = Image.open(data.file)
    img, _ = detector(img)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")
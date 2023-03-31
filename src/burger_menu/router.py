from fastapi import APIRouter, File, UploadFile, Response
from .schemas import BurgerMenu
from .utils import (
    convert_from_image_to_cv2,
    show_img,
)
from .detector import DetectorBurgerMenu

import base64
from io import BytesIO
from PIL import Image

router = APIRouter(
    prefix="/detect_burger_menu",
    tags=["Burger menu"],
)

detector = DetectorBurgerMenu()


@router.post("/")
async def get_burger_menu(data: BurgerMenu):
    """
    Детекция значка бургер-меню на полученном изображении
    """
    img = data.img_base64
    img = base64.b64decode(img)
    img = Image.open(BytesIO(img))
    # img = convert_from_image_to_cv2(img)
    img, elements = detector(img)

    # раскоментировать для отладки
    # show_img(convert_from_image_to_cv2(img))

    return {
        "status": "success",
        "data": elements,
        "details": None,
    }


@router.post("/test_json")
async def get_burger_menu(data: UploadFile = File(...)):
    """
    Тестирование детекции значка бургер-меню с возвращением координат
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
    Тестирование детекции значка бургер-меню с возвращением изображения
    """
    img = Image.open(data.file)
    img, _ = detector(img)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

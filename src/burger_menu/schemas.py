from pydantic import BaseModel


class BurgerMenu(BaseModel):
    img_base64: str

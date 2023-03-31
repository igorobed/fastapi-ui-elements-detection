from pydantic import BaseModel


class MenuList(BaseModel):
    img_before_base64: str
    img_after_base64: str

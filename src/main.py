import uvicorn
from fastapi import FastAPI

from src.burger_menu.router import router as router_burger_menu  # чтобы запускать из консоли из корня
from src.menu_items.router import router as router_menu_items
# from burger_menu.router import router as router_burger_menu
# from menu_items.router import router as router_menu_items


app = FastAPI(title="UI Elements Detection")

app.include_router(router_burger_menu)
app.include_router(router_menu_items)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

# fastapi-ui-elements-detection

## Запуск проекта:

### С помощью Docker:

1. cd fastapi-ui-elements-detection

2. docker build . -t fastapi_app

3. docker run -p 8000:8000 fastapi_app

4. Перейти по адресу: http://localhost:8000/docs


### Без Docker

1. cd fastapi-ui-elements-detection

2. pip install -r requirements/dev.txt

3. uvicorn src.main:app

4. 4. Перейти по адресу: http://localhost:8000/docs


FROM python:3.9.12-slim

WORKDIR /app
COPY . /app/

RUN pip install -r requirements.txt

EXPOSE 4242

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:4242", "predict:app" ]

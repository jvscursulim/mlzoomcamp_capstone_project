import requests

img = {"image_path": "/workspaces/mlzoomcamp_capstone_project/imgs/00001.jpg"}

url = "http://localhost:4242/predict"
print(requests.post(url, json=img).json())

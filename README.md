# MLZoomcamp 2022 Capstone Project

## A Convolutional Neural Network to detect cracks on a surface

### Description:

Concrete is one of the most used and important materials in civil construction. Our houses, our offices, the bridges that connect cities and a lot of other types of constructions, are made of concrete, or if concrete is not the main material, at least is used in some part of the construction. The major defect detected in civil structures are the surface cracks on concrete. It is important to make inspections in the structure to evaluate the rigidity and the tensile, once these concrete surface cracks can compromise the structure and cause big problems and disasters. Then it is very important to make inspections regularly in the structures in order to detect these cracks in the early stage to avoid major problems in the future. A convolutional neural network could be used to do a classification and an analysis if there is a surface crack in the photos taken during the process of structure inspection.

### How to run the application:
### 1. Docker

Step 1: Clone the GitHub repository with the project.
```bash
git clone https://github.com/jvscursulim/mlzoomcamp_capstone_project
```

Step 2: Access the GitHub repository folder.
```bash
cd mlzoomcamp_capstone_project
```

Step 3: Create the docker image.
```bash
docker build -t surface_crack_detection .
```

Step 4: Run the application with docker.
```bash
docker run -p 4242:4242 surface_crack_detection
```

Step 5: Put some images that you want to classify in imgs folder

Step 6: Change the value associated with the key `image_path` of the dictionary assigned with the variable `img` inside `test.py` file. The new value must be the path of one of the images inside imgs folder. (Please refer to section "How to send data for the application")

### 2. Without Docker

Step 1: Clone the GitHub repository with the project.
```bash
git clone https://github.com/jvscursulim/mlzoomcamp_capstone_project
```

Step 2: Access the GitHub repository folder.
```bash
cd mlzoomcamp_capstone_project
```

Step 3: Create a virtual environment.
```bash
python -m venv env
```

Step 4: Activate your virtual environment.
* Linux: Activation of the virtual environment.
```bash
source env/bin/activate
```

* Windows: Activation of the virtual environment.
```bash
env/Scripts/Activate.ps1
```

Step 5: Install pipenv.
```bash
pip install pipenv
```

Step 6: Install the packages required for this application using the command below.
```bash
pipenv install
```

Step 7: Run the application with gunicorn.
```bash
gunicorn --bind=0.0.0.0:4242 predict:app
```

Step 8: Put some images that you want to classify in imgs folder

Step 9: Change the value associated with the key `image_path` of the dictionary assigned with the variable `img` inside `test.py` file. The new value must be the path of one of the images inside imgs folder. (Please refer to section "How to send data for the application")

#### Observation: If you want to train a model

Access script folder.
```bash
cd script
```
Make your changes in `train.py` and run the file using the command below.
```bash
python train.py
```
After training process, you can build the new application following the instructions in the sections 1. Docker and 2. Without Docker.

### How to send data for the application:

For instance the `test.py` file.
### Code snippet

```python
import requests

img = {"image_path": "/workspaces/mlzoomcamp_capstone_project/imgs/00001.jpg"}

url = "http://localhost:4242/predict"
print(requests.post(url, json=img).json())
```

### References:

1. [MLZoomcamp Course Material - Session #8 - Deep Learning](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/08-deep-learning)
2. [Surface Crack Detection Dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)

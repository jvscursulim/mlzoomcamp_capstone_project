import numpy as np
import matplotlib.pyplot as plt
import tensorflow.lite as tflite
from flask import Flask, jsonify, request
from skimage.transform import resize

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    
    interpreter = tflite.Interpreter(model_path="model/cnn_surface_crack_detection_bs64_epochs5_lr_0_001.tflite")
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
        
    client = request.get_json()

    img = plt.imread(client["image_path"])
    img = resize(img, (150, 150))
    X = img
    X = X.reshape(1, 150, 150, 3)
    X = np.float32(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    
    result = {"concrete_surface_crack_probability": float(preds[0][0])}
    
    return jsonify(result)

if __name__ == "__main__":
    
    app.run(debug=True, host="0.0.0.0", port=4242)


import tensorflow.lite as tflite
from flask import Flask 

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    
    with open(file="", mode="r") as model_file:
        
        pass
        
    client = request.get_json()
    
    result = {}
    
    return jsonify(result)

if __name__ == "__main__":
    
    app.run(debug=True, host="0.0.0.0", port=4242)


from flask import Flask, request, jsonify
import torch
from yolov5 import YOLOv5

app = Flask(__name__)

# Load YOLOv5 model
model = YOLOv5('yolov5s.pt', device='cpu')  # Change to your model file

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    img_bytes = file.read()

    # Perform inference
    results = model(img_bytes)
    return jsonify(results)  # Return predictions

if __name__ == '__main__':
    app.run(debug=True)

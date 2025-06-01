from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import numpy as np
import base64
import uuid
import tempfile

app = Flask(__name__)
model = YOLO("best5.pt")

# Configure folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # Clear previous results on page reload
        return render_template("index.html")

    # Handle POST request
    image_path = None
    label = None
    message = None
    error = None

    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No image selected")

    try:
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png']:
            return render_template("index.html", error="Invalid file format. Please upload JPG, JPEG or PNG.")

        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Verify the image was saved properly
        if not os.path.exists(filepath):
            return render_template("index.html", error="Failed to save image")

        # Run prediction
        results = model.predict(source=filepath, save=False, show=False)

        if not results or len(results) == 0:
            raise ValueError("No results from model prediction")

        original_img = results[0].orig_img.copy()
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            names = results[0].names

            best_idx = confs.argmax()
            best_box = coords[best_idx]
            best_class_id = int(class_ids[best_idx])
            best_conf = confs[best_idx]

            x1, y1, x2, y2 = map(int, best_box)
            label = f"{names[best_class_id]}"

            # Draw bounding box and label
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(original_img, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
            cv2.putText(original_img, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            result_filename = f"result_{unique_filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
            image_path = result_path
        else:
            # No detection case
            label = "No Detection"
            message = "No cow detected. Please provide a clearer image of the cow's body."

            # Add text to image
            text = "No Cow Detected"
            font_scale = 2
            thickness = 4
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (original_img.shape[1] - text_width) // 2
            y = (original_img.shape[0] + text_height) // 2
            cv2.putText(original_img, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

            suggestion = "Please provide a clearer image"
            font_scale_small = 0.8
            thickness_small = 2
            (sugg_width, sugg_height), _ = cv2.getTextSize(suggestion, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small,
                                                           thickness_small)
            x_sugg = (original_img.shape[1] - sugg_width) // 2
            y_sugg = y + text_height + 20
            cv2.putText(original_img, suggestion, (x_sugg, y_sugg),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 0, 255), thickness_small)

            result_filename = f"result_{unique_filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
            image_path = result_path

    except Exception as e:
        error = f"Error processing image: {str(e)}"
        # Clean up files if they exist
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        if 'result_path' in locals() and os.path.exists(result_path):
            os.remove(result_path)
        return render_template("index.html", error=error)

    return render_template("index.html", image_path=image_path, label=label, message=message, error=error)


@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Process image
    img_bytes = file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run prediction
    results = model.predict(source=img, save=False, show=False)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        coords = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        names = results[0].names

        best_idx = confs.argmax()
        best_box = coords[best_idx]
        best_class_id = int(class_ids[best_idx])
        best_conf = confs[best_idx]

        x1, y1, x2, y2 = map(int, best_box)
        label = f"{names[best_class_id]}"

        # Enhanced bounding box and label
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(img_rgb, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
        cv2.putText(img_rgb, label, (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Convert back to JPEG
    _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({
        'image': img_base64,
        'label': label if boxes is not None and len(boxes) > 0 else "No detection",
        'has_detection': boxes is not None and len(boxes) > 0
    })

if __name__ == '__main__':
    app.run(debug=True)
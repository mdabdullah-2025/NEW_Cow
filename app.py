from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# --- Model Loading ---
# Load the YOLO model from the ONNX file.
# Ensure 'best6.onnx' is in the same directory as this Flask app,
# or provide the full path to the file.
# Explicitly setting task='detect' helps Ultralytics avoid guessing
# and can prevent some warnings.
try:
    model = YOLO("best6.onnx", task='detect')
    print("Model loaded successfully from best6.onnx")
except Exception as e:
    print(f"ERROR: Failed to load model from best6.onnx. Please ensure the file exists and is valid. Details: {e}")
    # In a production environment, you might want to exit or return a specific error page
    model = None # Set model to None if loading fails, to prevent further errors

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main page, allowing users to upload an image for cow detection.
    Processes the uploaded image, runs inference, and displays results.
    """
    if request.method == 'GET':
        return render_template("index.html")

    # Handle POST request for image upload
    image_data = None
    label = None
    message = None
    error = None

    # Check if a file was uploaded in the request
    if "image" not in request.files:
        return render_template("index.html", error="No image part in the request. Please upload an image.")

    file = request.files["image"]
    # Check if the file input was empty
    if file.filename == "":
        return render_template("index.html", error="No image selected. Please choose an image file.")

    # Check if the model was loaded successfully at application startup
    if model is None:
        return render_template("index.html", error="Server error: Model failed to load at startup. Please contact support.")

    try:
        # Read image bytes directly into a NumPy array, then decode with OpenCV
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return render_template("index.html", error="Invalid image file. Could not decode the uploaded file.")

        # Run prediction on the image.
        # The 'source' argument can directly take a numpy array.
        # save=False and show=False prevent Ultralytics from writing files or displaying windows.
        results = model.predict(source=img, save=False, show=False)

        # Ensure results are not empty
        if not results or len(results) == 0:
            raise ValueError("Model prediction returned no results. This might indicate an issue with the model or input image.")

        # Get the original image from results and convert to RGB for consistent drawing
        original_img = results[0].orig_img.copy()
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        boxes = results[0].boxes # Access the bounding boxes object
        if boxes is not None and len(boxes) > 0:
            # Extract confidence scores, bounding box coordinates, and class IDs
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            names = results[0].names # Get class names from the model results

            # Find the detection with the highest confidence score
            best_idx = confs.argmax()
            best_box = coords[best_idx]
            best_class_id = int(class_ids[best_idx])
            best_conf = confs[best_idx]

            # Extract coordinates and format the label
            x1, y1, x2, y2 = map(int, best_box)
            label = f"{names[best_class_id]} ({best_conf:.2f})" # Include confidence in label

            # Draw bounding box on the image
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 4) # Green rectangle, 4px thickness
            
            # Prepare text for the label background (to ensure readability)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            # Draw a filled rectangle as background for the label
            cv2.rectangle(original_img, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1) # Green filled rectangle
            # Put the label text on the image
            cv2.putText(original_img, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3) # White text, 3px thickness

        else:
            # Case where no cow is detected
            label = "No Detection"
            message = "No cow detected in the image. Please provide a clearer image of the cow's body."

            # Add "No Cow Detected" text to the image for visual feedback
            text = "No Cow Detected"
            font_scale = 2
            thickness = 4
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (original_img.shape[1] - text_width) // 2
            y = (original_img.shape[0] + text_height) // 2
            cv2.putText(original_img, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness) # Red text

            # Add a suggestion below the main text
            suggestion = "Please provide a clearer image"
            font_scale_small = 0.8
            thickness_small = 2
            (sugg_width, sugg_height), _ = cv2.getTextSize(suggestion, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small,
                                                           thickness_small)
            x_sugg = (original_img.shape[1] - sugg_width) // 2
            y_sugg = y + text_height + 20 # Position below the main text
            cv2.putText(original_img, suggestion, (x_sugg, y_sugg),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 0, 255), thickness_small) # Red text

        # Convert the processed image (which is currently in RGB) back to BGR for OpenCV encoding,
        # then encode it to JPEG format and finally convert to base64 for embedding in HTML.
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        image_data = base64.b64encode(img_encoded).decode('utf-8')

    except Exception as e:
        # Catch any exceptions during image processing or prediction
        error = f"An unexpected error occurred during image processing: {str(e)}"
        print(f"ERROR in index route: {e}") # Log the full error for debugging
        return render_template("index.html", error=error)

    return render_template("index.html", image_data=image_data, label=label, message=message, error=error)


@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    """
    Handles real-time prediction requests, typically from a webcam stream or frequent image uploads.
    Receives an image, runs inference, and returns the processed image and detection status as JSON.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded in the request.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for real-time prediction.'}), 400

    if model is None:
        return jsonify({'error': 'Server error: Model failed to load at startup.'}), 500

    try:
        # Process image from the request
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data received for real-time prediction.'}), 400

        # Convert to RGB for consistent processing and drawing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run prediction
        results = model.predict(source=img, save=False, show=False)

        label = "No detection" # Default label if no cow is found
        has_detection = False
        
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            has_detection = True
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            names = results[0].names

            best_idx = confs.argmax()
            best_box = coords[best_idx]
            best_class_id = int(class_ids[best_idx])
            best_conf = confs[best_idx]

            x1, y1, x2, y2 = map(int, best_box)
            label = f"{names[best_class_id]} ({best_conf:.2f})"

            # Draw bounding box and label on the RGB image
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(img_rgb, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
            cv2.putText(img_rgb, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Convert the processed RGB image back to BGR for JPEG encoding, then to base64
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({
            'image': img_base64,
            'label': label,
            'has_detection': has_detection
        })

    except Exception as e:
        # Catch any exceptions during real-time processing
        print(f"ERROR in predict_realtime route: {e}") # Log the full error
        return jsonify({'error': f'An unexpected error occurred during real-time prediction: {str(e)}'}), 500


if __name__ == '__main__':
    # Ensure the 'templates' directory exists for render_template to find index.html
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Run the Flask app in debug mode for local development.
    # For production, Gunicorn will manage the app.
    app.run(debug=True, host='0.0.0.0', port=5000)

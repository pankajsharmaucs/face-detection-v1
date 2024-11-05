from flask import Flask, render_template, Response, jsonify
import cv2
import threading

app = Flask(__name__)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables for controlling the camera
camera_active = False
cap = None  # Camera capture object
camera_index = 0  # Start with the first camera
camera_lock = threading.Lock()  # Lock for thread safety

def check_camera(index):
    """Check if a camera at the given index is accessible."""
    test_cap = cv2.VideoCapture(index)
    is_opened = test_cap.isOpened()
    test_cap.release()  # Release the camera
    return is_opened

def generate_frames():
    global cap
    with camera_lock:
        # Start the camera only if it's not already active
        if cap is None:
            cap = cv2.VideoCapture(camera_index)

    while camera_active:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    with camera_lock:
        if cap is not None:
            cap.release()
            cap = None  # Release the camera resource

@app.route('/video_feed')
def video_feed():
    if not camera_active:
        return Response(status=204)  # No content if camera is not active
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    camera_active = True
    print("Camera started")  # Debugging line
    return jsonify({'status': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, cap
    camera_active = False  # Set the flag to False to stop the stream
    print("Camera stopped")  # Debugging line
    with camera_lock:
        if cap is not None:
            cap.release()  # Release the camera
            cap = None  # Set the capture object to None
    return jsonify({'status': 'Camera stopped'})

@app.route('/change_camera', methods=['POST'])
def change_camera():
    global cap, camera_index, camera_active
    new_camera_index = (camera_index + 1) % 2  # Change camera index (0 or 1)

    if check_camera(new_camera_index):
        with camera_lock:
            if cap is not None:
                cap.release()
            camera_active = False  # Stop current feed
            camera_index = new_camera_index  # Change to the new camera index
            camera_active = True  # Restart feed
        print(f"Switched to camera {camera_index}")  # Debugging line
        return jsonify({'status': f'Camera changed to {camera_index}'})
    else:
        print(f"Camera {new_camera_index} not available")  # Debugging line
        return jsonify({'status': f'Camera {new_camera_index} not available'}), 400

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

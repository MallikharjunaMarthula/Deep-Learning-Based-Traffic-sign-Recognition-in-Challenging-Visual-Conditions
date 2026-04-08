from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load trained model
model = YOLO("runs/detect/traffic_signs_kaggle2/weights/best.pt")

# 🔥 Replace with your mobile IP
ip_camera_url = "http://192.0.0.4:8080/video"
cap = cv2.VideoCapture(ip_camera_url)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)[0]
        annotated_frame = results.plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

print("Loaded model:", model.names)
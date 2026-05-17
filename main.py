import os
import csv
import time
from datetime import datetime

from flask import Flask, Response, render_template_string

# ==============================
# SAFE IMPORTS
# ==============================
try:
    import cv2
except:
    cv2 = None

try:
    import numpy as np
except:
    np = None

try:
    import mediapipe as mp
except:
    mp = None

# face_recognition often crashes Render free plan
try:
    import face_recognition
except:
    face_recognition = None

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)

# ==============================
# HTML TEMPLATE
# ==============================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Attendance</title>

    <style>
        body{
            background:#111;
            color:white;
            text-align:center;
            font-family:Arial;
        }

        img{
            width:80%;
            border:5px solid white;
            margin-top:20px;
        }

        h1{
            margin-top:20px;
        }
    </style>
</head>

<body>

<h1>Smart Attendance System</h1>

<img src="/video_feed">

</body>
</html>
"""

# ==============================
# LOAD KNOWN FACES
# ==============================
known_encodings = []
known_names = []

folder = "known_faces"

if not os.path.exists(folder):
    os.makedirs(folder)

if face_recognition is not None:

    files = os.listdir(folder)

    for file in files:

        if file.endswith(".jpg") or file.endswith(".png"):

            path = os.path.join(folder, file)

            try:
                image = face_recognition.load_image_file(path)

                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:

                    known_encodings.append(encodings[0])

                    known_names.append(
                        os.path.splitext(file)[0]
                    )

            except Exception as e:
                print("Face Load Error:", e)

print("Loaded Faces:", known_names)

# ==============================
# MEDIAPIPE
# ==============================
face_mesh = None

if mp is not None:

    try:
        mp_face_mesh = mp.solutions.face_mesh

        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=False
        )

    except Exception as e:
        print("MediaPipe Error:", e)

# ==============================
# CAMERA
# ==============================
cap = None

if cv2 is not None:

    try:
        cap = cv2.VideoCapture(0)

        if cap.isOpened():

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            print("Camera Opened")

        else:
            cap = None

    except Exception as e:
        print("Camera Error:", e)
        cap = None

# ==============================
# ATTENDANCE FILE
# ==============================
attendance_file = "attendance.csv"

marked_students = set()

# ==============================
# ATTENDANCE FUNCTION
# ==============================
def mark_attendance(name):

    if name in marked_students:
        return

    now = datetime.now()

    dt = now.strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.isfile(attendance_file)

    with open(attendance_file, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Time", "Status"])

        writer.writerow([name, dt, "Present"])

    marked_students.add(name)

    print(name, "marked present")

# ==============================
# VIDEO FRAMES
# ==============================
def gen_frames():

    # If OpenCV missing
    if cv2 is None or np is None:

        while True:

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + b''
                + b'\r\n'
            )

            time.sleep(1)

    # If camera not available
    if cap is None:

        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2.putText(
            blank,
            "Camera Not Available On Render",
            (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        while True:

            ret, buffer = cv2.imencode(".jpg", blank)

            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + frame
                + b'\r\n'
            )

            time.sleep(1)

    # Normal webcam stream
    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        # ==========================
        # FACE RECOGNITION
        # ==========================
        if face_recognition is not None:

            try:
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                rgb_small = cv2.cvtColor(
                    small,
                    cv2.COLOR_BGR2RGB
                )

                face_locations = face_recognition.face_locations(
                    rgb_small
                )

                face_encodings = face_recognition.face_encodings(
                    rgb_small,
                    face_locations
                )

                face_locations = [
                    (t * 2, r * 2, b * 2, l * 2)
                    for (t, r, b, l) in face_locations
                ]

                for (top, right, bottom, left), face_encoding in zip(
                    face_locations,
                    face_encodings
                ):

                    name = "Unknown"

                    matches = face_recognition.compare_faces(
                        known_encodings,
                        face_encoding
                    )

                    if True in matches:

                        match_index = matches.index(True)

                        name = known_names[match_index]

                        mark_attendance(name)

                    cv2.rectangle(
                        frame,
                        (left, top),
                        (right, bottom),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        name,
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            except Exception as e:
                print("Recognition Error:", e)

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame
            + b'\r\n'
        )

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/video_feed")
def video_feed():

    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
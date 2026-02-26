import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import time

from utils.video import camera_stream
from utils.detector import detect_object
from utils.morse import text_to_morse

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")

MAX_FPS = 12
DETECT_EVERY_N_FRAMES = 6
JPEG_QUALITY = 70

last_label = None
last_morse = None


def generate_frames():
    global last_label, last_morse

    frame_count = 0

    for frame in camera_stream():
        start_time = time.time()
        frame_count += 1

        label = last_label or "Detecting..."
        morse = last_morse or ""

        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            detected_label, bbox = detect_object(frame)
            if detected_label:
                label = detected_label
                morse = text_to_morse(label)
                last_label = label
                last_morse = morse

                socketio.emit(
                    "update_caption",
                    {"label": label, "morse": morse}
                )

        h, w, _ = frame.shape

        cv2.rectangle(frame, (0, h - 110), (w, h), (0, 0, 0), -1)

        cv2.putText(
            frame, "Detected Object:", (20, h - 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            frame, label, (260, h - 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame, "Morse Code:", (20, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            frame, morse, (260, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        ret, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        elapsed = time.time() - start_time
        time.sleep(max(0, (1 / MAX_FPS) - elapsed))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", debug=False)


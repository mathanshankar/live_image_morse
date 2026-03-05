import cv2
import time


def camera_stream():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    while True:
        start = time.time()

        success, frame = cap.read()
        if not success:
            break

        yield frame

        elapsed = time.time() - start
        time.sleep(max(0, (1 / 15) - elapsed))

    cap.release()

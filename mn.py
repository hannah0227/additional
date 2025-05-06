import cv2
from flask import Flask, Response
import RPi.GPIO as GPIO
import time
import numpy as np

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

def initialize_tracker():
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=False
    )
    return deepsort

GPIO.setmode(GPIO.BCM)
LED_pin = 2
GPIO.setup(LED_pin, GPIO.OUT)
GPIO.output(LED_pin, GPIO.LOW)

app = Flask(__name__)

MODEL_PB = "path/to/your/ssd_mobilenet_v2_coco_2018_03_29.pb"
MODEL_PBTXT = "path/to/your/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
CLASSES_FILE = "path/to/your/coco.names"

try:
    net = cv2.dnn.readNetFromTensorflow(MODEL_PB, MODEL_PBTXT)
except Exception as e:
    print(f"Error loading SSD-MobileNet model: {e}")
    print("Please check the paths to your .pb and .pbtxt files.")
    exit()

try:
    with open(CLASSES_FILE, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Error loading class names file: {e}")
    print("Please check the path to your class names file.")
    exit()

TARGET_CLASSES = ['car', 'motorcycle', 'bicycle']
TARGET_CLASS_IDS = [CLASSES.index(cls) for cls in TARGET_CLASSES if cls in CLASSES]
print(f"Target Class Names: {TARGET_CLASSES}")
print(f"Corresponding Class IDs: {TARGET_CLASS_IDS}")

CONFIDENCE_THRESHOLD = 0.3

tracker = initialize_tracker()

cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open the camera device /dev/video2")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
         print("Cannot open default camera (index 0) either. Exiting.")
         exit()
    print("Opened default camera (index 0) instead.")

def generate_frames():
    object_detected_this_cycle = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, retrying...")
                time.sleep(0.1)
                continue

            h, w, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()

            current_frame_detections = []
            current_frame_has_target = False

            detections = detections[0, 0]

            for detection in detections:
                confidence = detection[2]
                class_id = int(detection[1])

                if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASSES):
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)

                    current_frame_detections.append([x1, y1, x2, y2, confidence, class_id])

                    if class_id in TARGET_CLASS_IDS:
                         current_frame_has_target = True

            if current_frame_detections:
                 detections_np = np.array(current_frame_detections)
                 tracked_objects = tracker.update(detections_np)
            else:
                 tracked_objects = tracker.update(np.empty((0, 6)))

            if current_frame_has_target:
                if not object_detected_this_cycle:
                    GPIO.output(LED_pin, GPIO.HIGH)
                    object_detected_this_cycle = True
            else:
                if object_detected_this_cycle:
                    GPIO.output(LED_pin, GPIO.LOW)
                    object_detected_this_cycle = False

            annotated_frame = frame.copy()

            for track in tracked_objects:
                 bbox = track.to_tlbr()
                 track_id = track.track_id
                 class_id = track.class_id
                 confidence = track.det_conf

                 x1, y1, x2, y2 = map(int, bbox)

                 class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"Class {class_id}"

                 color = (0, 255, 0)

                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                 label = f"{class_name} ID: {track_id}"
                 (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                 cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
                 cv2.putText(annotated_frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Failed to encode frame.")
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        print("generate_frames generator finished or an error occurred.")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Real-time SSD-MobileNet & DeepSORT Tracking</title>
      </head>
      <body>
        <h1>Live Camera Feed with SSD-MobileNet & DeepSORT Tracking</h1>
        <img src="/video_feed" width="640" height="480">
      </body>
    </html>
    """

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        if cap.isOpened():
            cap.release()
            print("Camera released.")
        GPIO.cleanup()
        print("GPIO cleaned up.")
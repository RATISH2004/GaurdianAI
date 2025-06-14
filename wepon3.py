import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from face_matcher import FaceMatcher

# Configuration
VIDEO_PATH = r"C:/Users/ratis/OneDrive\Desktop/NUS application/GAIP/FireArms-model-training/input/sp3.mp4"  # <-- Change this
PATH_TO_SAVED_MODEL = "D:/ML_Training/workspace/training_demo/exported-models/my_model_v1/saved_model"
PATH_TO_LABELMAP = "D:/ML_Training/workspace/training_demo/annotations/label_map.pbtxt"
SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load weapon detection model and label map
print("🔁 Loading model...")
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELMAP, use_display_name=True)

# Initialize face matcher
face_matcher = FaceMatcher("./known_faces")

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Failed to open video: {VIDEO_PATH}")
    exit()

print("✅ Processing video... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    face_detections = face_matcher.process_frame(frame)
    for detection in face_detections:
        name = detection['name']
        matches = detection['matches']
        print(f"🧑 Face match: {name} (Matches: {matches})")
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Detect weapons
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (320, 320))
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame_resized, 0), dtype=tf.uint8)

    detections = model(input_tensor)
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    weapon_detected = False
    for i in range(len(scores)):
        if scores[i] > 0.45 and category_index[classes[i]]['name'].lower() == 'guns':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"🔫 Weapon detected! Frame saved: {filename}")
            weapon_detected = True
            break

    # Visualize detections
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.45,
        line_thickness=2)

    cv2.imshow('Gun & Face Detection - Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Video processing completed.")

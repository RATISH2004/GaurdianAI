import os
import time
import threading
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from imagehash import dhash

# Configuration
INPUT_DIR = "detected_frames"
KNOWN_DIR = "known_faces"
MIN_FACE_CONFIDENCE = 0.3    # YOLOv8 detection threshold
DEDUPE_INTERVAL = 60        # 60 seconds between cleanups
SIMILARITY_THRESHOLD = 23     # 0-64 (lower = stricter)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(KNOWN_DIR, exist_ok=True)

# Load YOLOv8 face detection model
try:
    yolo_model = YOLO("./face_model/yolov8n_face.pt")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {str(e)}")
    exit(1)

class FaceExtractHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(('.jpg', '.png')):
            return
        
        print(f"📥 New frame: {os.path.basename(event.src_path)}")
        threading.Thread(target=self.process_frame, args=(event.src_path,)).start()

    def process_frame(self, path):
        # Retry reading for file locks
        img = None
        for _ in range(3):
            try:
                img = cv2.imread(path)
                if img is not None and img.size > 0:
                    break
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Read error: {str(e)}")
        
        if img is None:
            print(f"❌ Failed to process: {os.path.basename(path)}")
            return

        try:
            results = yolo_model(img, conf=MIN_FACE_CONFIDENCE)[0]
            print(f"🔍 Detected {len(results.boxes)} faces in frame")
            
            for i, box in enumerate(results.boxes):
                conf = box.conf.item()
                print(f"  Face {i+1}: confidence={conf:.2f}")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                face_crop = img[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                
                # Save logic
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                save_path = os.path.join(KNOWN_DIR, f"face_{timestamp}_{i}.jpg")
                cv2.imwrite(save_path, face_crop)
                print(f"✅ Saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"❌ Processing error: {str(e)}")

def safe_deduplicate():
    """Remove duplicates while keeping the first 3 occurrences"""
    print("🧹 Starting safe deduplication (keep 3)...")
    hash_groups = {}  # {base_hash: [file_paths]}
    
    # Group similar faces
    for fname in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, fname)
        if not os.path.isfile(path):
            continue
            
        try:
            with Image.open(path) as img:
                current_hash = dhash(img)
                matched = False
                
                # Find existing similar group
                for existing_hash in hash_groups:
                    if (current_hash - existing_hash) <= SIMILARITY_THRESHOLD:
                        hash_groups[existing_hash].append(path)
                        matched = True
                        break
                        
                if not matched:
                    hash_groups[current_hash] = [path]
                    
        except Exception as e:
            print(f"⚠️ Hash error: {os.path.basename(path)} - {str(e)}")
    
    # Process groups
    to_delete = []
    for group in hash_groups.values():
        # Sort by creation time (oldest first)
        group.sort(key=lambda x: os.path.getctime(x))
        
        # Keep max 3 oldest, mark rest for deletion
        if len(group) > 3:
            to_delete.extend(group[3:])

    # Delete excess files
    for path in to_delete:
        try:
            os.remove(path)
            print(f"🗑 Removed duplicate: {os.path.basename(path)}")
        except Exception as e:
            print(f"⚠️ Deletion failed: {os.path.basename(path)} - {str(e)}")

    print(f"✅ Deduplication complete. Kept 3 samples per face group.")


def maintenance_loop():
    """Periodic cleanup with safety checks"""
    while True:
        time.sleep(DEDUPE_INTERVAL)
        safe_deduplicate()

print("🚀 Face Manager v1.0")
print(f"📂 Input: {os.path.abspath(INPUT_DIR)}")
print(f"📦 Storage: {os.path.abspath(KNOWN_DIR)}")
print(f"🔍 Similarity threshold: {SIMILARITY_THRESHOLD} bits")
    
    # Setup observer
observer = Observer()
observer.schedule(FaceExtractHandler(), INPUT_DIR, recursive=False)
observer.start()

    # Start maintenance thread
threading.Thread(target=maintenance_loop, daemon=True).start()

try:
        while observer.is_alive():
            time.sleep(1)
except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Shutting down...")
    
observer.join()
print("✅ System stopped safely")

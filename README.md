# GuardianAI
CCTV based firearm detection and suspect tracking



**Guardian AI** is an integrated real-time face and weapon detection system designed for surveillance and security applications. It combines state-of-the-art deep learning (YOLOv8) for face and weapon detection with fast template-based face matching (ORB), providing robust, efficient monitoring and alerting.

---

## Features

- **Real-time Face Detection** using YOLOv8n-face
- **Weapon Detection** (e.g., guns) using a custom-trained YOLO/SSD model
- **Template-based Face Matching** using ORB for fast, lightweight recognition
- **Duplicate Face Filtering** using perceptual hashing (dhash)
- **Automatic Frame Saving** on detection events
- **Directory Monitoring** for new frames and face templates
- **Periodic Maintenance** to remove duplicate faces and manage storage
- **Easy Extensibility** for new detection classes or recognition methods

---

## Directory Structure

```
GAudian-AI/
│
├── detected_frames/         # Incoming frames for face/weapon detection
├── known_faces/             # Cropped and deduplicated face images
├── face_model/              # YOLOv8n-face model file
├── face_matcher.py          # ORB-based face matcher
├── wepon3.py                # Main video processing script (face + weapon)
├── requirements.txt         # Python dependencies
├──face_extractor.py         # To extract faces (to be run simulatneously)
├── README.md                # This file
└── ...                      # Other scripts and resources
```


---

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-org/GAudian-AI.git
cd GAudian-AI
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Download Models**
    - Place your YOLOv8n-face model (`yolov8n_face.pt`) in the `face_model/` directory.
    - Place your weapon detection model in the appropriate path as referenced in your scripts.
4. **Prepare Directories**
    - Ensure `detected_frames/` and `known_faces/` directories exist (scripts will create them if missing).

---

## How It Works

### 1. **Video Processing (`wepon3.py`)**

- Loads a video and processes each frame.
- **Weapon Detection:**
Uses a TensorFlow SSD/YOLO model to detect weapons (e.g., guns). On detection, saves the frame.
- **Face Detection:**
Uses YOLOv8n-face to detect faces in each frame.
- **Face Matching:**
Passes detected faces to `FaceMatcher` (ORB-based) to match against known faces.


### 2. **Face Extraction and Deduplication**

- **Face Cropping:**
Detected faces are cropped and saved to `known_faces/`.
- **Deduplication:**
Uses perceptual hashing (dhash) to group and remove duplicate faces, keeping only the oldest 3 per group.


### 3. **Face Matching (`face_matcher.py`)**

- Loads face templates from `known_faces/`.
- Uses ORB to detect and describe keypoints, and matches them using Hamming distance.
- Robust to rotation, efficient for real-time use.


### 4. **Directory Monitoring**

- Uses `watchdog` to monitor directories for new frames or templates, triggering processing automatically.

---

## Key Technologies

- **YOLOv8n-face:** Deep learning model for fast, accurate face detection.
- **TensorFlow SSD/YOLO:** For weapon detection.
- **ORB (Oriented FAST and Rotated BRIEF):** Fast, rotation-invariant template matching for faces.
- **Perceptual Hashing (dhash):** For duplicate face detection.
- **OpenCV:** Image processing and computer vision operations.
- **Watchdog:** Real-time directory monitoring.
- **Python threading:** For concurrent processing and maintenance.

---
## Model Training
refer transfer learning guide([https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#]) to train your custom model
## Usage

1. **Start the main video processing script**

```bash
python wepon3.py
```

    - The script will display detections and save frames with detected weapons/faces.
2. **Face Extraction and Deduplication**
    - The system automatically extracts, saves, and deduplicates faces from new frames.
3. **Face Matching**
    - New faces are matched against `known_faces/` using ORB.

---

## Configuration

- **Detection Thresholds:**
    - `MIN_FACE_CONFIDENCE` for face detection (default: 0.3)
    - `SIMILARITY_THRESHOLD` for face deduplication (default: 23)
- **Directories:**
    - `INPUT_DIR`, `KNOWN_DIR` can be set in the scripts.
- **Model Paths:**
    - Update paths to your YOLO/SSD models as needed.

---

## Notes \& Limitations

- **ORB-based face matching is fast and license-free, but less robust than deep learning-based face recognition for unconstrained environments.**
- **For best results, use high-quality, well-lit face templates.**
- **Weapon detection accuracy depends on the quality and training of your detection model.**
- **All detection and matching is done on the CPU by default; GPU acceleration is possible with compatible hardware and libraries.**

---

## Example Output

- Real-time video window with bounding boxes for faces and weapons.
- Console logs for detection events and deduplication status.
- Cropped face images stored in `known_faces/`, limited to 3 per unique face.

---

## Authors

- GunShy Coders
      Ratish R A
      Ashman Sodhi
      Adway Aghor
      Aniket Singh
      Chaarulatha
      Shreya Samridhi
- GAIP June 2025, Corporate Gurukul, National University of Singapore


---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- YOLOv8 by Ultralytics
- OpenCV community
- TensorFlow Object Detection API
- ([https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#]) 

---

**Guardian AI — Guarding with Intelligence.**



# face_matcher.py
import cv2
import os
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FaceMatcher:
    def __init__(self, template_dir, threshold=0.7):
        self.template_dir = template_dir
        self.threshold = threshold
        self.templates = self._load_templates()
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Set up directory watcher
        self.event_handler = TemplateHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, template_dir, recursive=False)
        self.observer.start()

    def _load_templates(self):
        templates = {}
        for fname in os.listdir(self.template_dir):
            path = os.path.join(self.template_dir, fname)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[fname] = img
        return templates

    def update_templates(self):
        self.templates = self._load_templates()

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = self.orb.detectAndCompute(gray, None)
        detections = []
        
        # Check if frame descriptors are valid
        if desc_frame is None:
            return detections

        for name, template in self.templates.items():
            kp_temp, desc_temp = self.orb.detectAndCompute(template, None)
            # Check if template descriptors are valid
            if desc_temp is None:
                continue

            # Check descriptor compatibility (dtype and shape)
            if desc_temp.dtype != desc_frame.dtype or desc_temp.shape[1] != desc_frame.shape[1]:
                continue

            matches = self.bf.match(desc_temp, desc_frame)
            if len(matches) > 15:
                good_matches = sorted(matches, key=lambda x: x.distance)
                src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                
                # Find homography and bounding box
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = template.shape
                    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, M)
                    x, y, w, h = cv2.boundingRect(dst)
                    detections.append({
                        'name': os.path.splitext(name)[0],
                        'bbox': (x, y, x+w, y+h),
                        'matches': len(good_matches)
                    })
        
        return detections


class TemplateHandler(FileSystemEventHandler):
    def __init__(self, matcher):
        self.matcher = matcher
        
    def on_modified(self, event):
        if not event.is_directory:
            self.matcher.update_templates()
            
    def on_created(self, event):
        if not event.is_directory:
            self.matcher.update_templates()
            
    def on_deleted(self, event):
        if not event.is_directory:
            self.matcher.update_templates()

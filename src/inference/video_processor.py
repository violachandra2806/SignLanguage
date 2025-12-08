import cv2
import numpy as np
import tempfile
import requests
import os
import time
from ultralytics import YOLO

class SIBITranslator:
    def __init__(self, model_path='runs/detect/sibi_guaranteed/weights/best.pt'):
        """Initialize with your trained model"""
        print(f"📦 Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = ['Ayah', 'Halo', 'Kakak', 'Minum', 'Rumah', 'Sama-sama', 'Sehat', 'Teman', 'Terima kasih', 'Tidur', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        print(f"✅ SIBI Translator loaded with {len(self.class_names)} classes!")
    
    def download_video(self, video_url):
        """Download video from URL"""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        
        try:
            print(f"⬇️ Downloading video from: {video_url}")
            
            if video_url.startswith('file://'):
                # Local file
                local_path = video_url[7:]
                if os.path.exists(local_path):
                    import shutil
                    shutil.copy2(local_path, tmp.name)
                    print(f"📂 Using local file: {local_path}")
                    return tmp.name
                else:
                    raise Exception(f"Local file not found: {local_path}")
            
            # Download from URL
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(video_url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(tmp.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if percent % 20 < 1:  # Print every 20%
                                print(f"   Progress: {percent:.1f}%")
            
            print(f"✅ Video downloaded: {os.path.getsize(tmp.name) / 1024 / 1024:.2f} MB")
            return tmp.name
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise Exception(f"Failed to download video: {str(e)}")
    
    def extract_frames_at_intervals(self, video_path, interval_seconds=1.0):
        """Extract frames at regular intervals for analysis"""
        print(f"🎞️ Extracting frames from: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   FPS: {fps:.1f}, Frames: {total_frames}, Duration: {duration:.1f}s")
        
        frame_interval = max(1, int(fps * interval_seconds))
        
        frames_data = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames_data.append({
                    "timestamp": timestamp,
                    "frame": frame.copy(),
                    "frame_count": frame_count
                })
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"   Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        print(f"✅ Extracted {len(frames_data)} frames at {interval_seconds}s intervals")
        return frames_data, fps
    
    def process_frame(self, frame, conf_threshold=0.5):
        """Process a single frame and return detections"""
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence >= conf_threshold:
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                            detections.append({
                                "class": class_name,
                                "confidence": confidence,
                                "bbox": box.xyxy[0].tolist()
                            })
        
        # Return the highest confidence detection
        if detections:
            best_detection = max(detections, key=lambda x: x['confidence'])
            return best_detection['class'], best_detection['confidence']
        
        return None, 0.0
    
    def process_video_predictions(self, video_url, interval_seconds=1.0, conf_threshold=0.5):
        """Main function to process video and get predictions with timestamps"""
        video_path = None
        try:
            print("\n" + "="*50)
            print("🚀 STARTING SIBI VIDEO ANALYSIS")
            print("="*50)
            
            # Step 1: Download video
            video_path = self.download_video(video_url)
            
            # Step 2: Extract frames
            frames_data, fps = self.extract_frames_at_intervals(video_path, interval_seconds)
            
            if not frames_data:
                print("⚠️ No frames extracted from video")
                return []
            
            # Step 3: Process each frame
            predictions = []
            processed_frames = 0
            
            for frame_data in frames_data:
                timestamp = frame_data["timestamp"]
                frame = frame_data["frame"]
                
                # Run inference
                sign_class, confidence = self.process_frame(frame, conf_threshold)
                
                if sign_class and confidence > conf_threshold:
                    predictions.append({
                        "second": timestamp,
                        "text": sign_class,
                        "confidence": confidence,
                        "frame_count": frame_data["frame_count"]
                    })
                    print(f"⏱️ {timestamp:.1f}s: '{sign_class}' (conf: {confidence:.2f})")
                
                processed_frames += 1
            
            # Step 4: Group consecutive predictions
            grouped_predictions = self.group_consecutive_predictions(predictions)
            
            print("\n" + "="*50)
            print(f"📊 ANALYSIS COMPLETE")
            print(f"   Total frames processed: {processed_frames}")
            print(f"   Raw detections: {len(predictions)}")
            print(f"   Grouped predictions: {len(grouped_predictions)}")
            print("="*50 + "\n")
            
            return grouped_predictions
            
        except Exception as e:
            print(f"❌ Error in SIBI video processing: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        finally:
            # Clean up temporary file
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                    print(f"🧹 Cleaned up temporary file: {video_path}")
                except:
                    pass
    
    def group_consecutive_predictions(self, predictions, time_threshold=2.0):
        """Group consecutive same predictions"""
        if not predictions:
            return []
        
        predictions.sort(key=lambda x: x['second'])
        grouped = []
        
        i = 0
        while i < len(predictions):
            current = predictions[i]
            j = i + 1
            
            # Find consecutive same predictions
            while j < len(predictions) and \
                  predictions[j]['text'] == current['text'] and \
                  predictions[j]['second'] - predictions[i]['second'] <= time_threshold:
                j += 1
            
            # Get average timestamp and best confidence in this group
            group = predictions[i:j]
            avg_second = sum(p['second'] for p in group) / len(group)
            best_in_group = max(group, key=lambda x: x['confidence'])
            
            grouped.append({
                "second": avg_second,
                "text": best_in_group['text'],
                "confidence": best_in_group['confidence'],
                "duration": group[-1]['second'] - group[0]['second'] + 1.0
            })
            
            i = j
        
        return grouped

# Global instance
sibi_translator = SIBITranslator()

def get_sibi_translation(video_url):
    """Main function to get translation from local SIBI model"""
    return sibi_translator.process_video_predictions(
        video_url, 
        interval_seconds=1.0,  # Analyze every 1 second
        conf_threshold=0.5     # Confidence threshold
    )